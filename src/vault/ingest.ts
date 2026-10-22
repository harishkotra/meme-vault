/** Turns whatever the browser can decode into a thumbnail plus a normalised embed blob. */

const EMBED_MAX = 640;
const THUMB_MAX = 320;

export type Prepared = {
  thumb: Blob;
  thumbType: string;
  embed: Blob;
  width: number;
  height: number;
};

async function decode(file: Blob): Promise<ImageBitmap | HTMLImageElement> {
  try {
    return await createImageBitmap(file);
  } catch {
    // Fallback for SVG and other formats createImageBitmap refuses.
    const url = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      return image;
    } finally {
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    }
  }
}

function draw(
  source: ImageBitmap | HTMLImageElement,
  max: number,
): { canvas: HTMLCanvasElement; width: number; height: number } {
  const sw = 'width' in source ? source.width : 0;
  const sh = 'height' in source ? source.height : 0;
  const scale = Math.min(1, max / Math.max(sw, sh || 1));
  const width = Math.max(1, Math.round(sw * scale));
  const height = Math.max(1, Math.round(sh * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas 2D is unavailable.');
  context.imageSmoothingQuality = 'high';
  // Memes are often transparent PNGs; a flat surface keeps the embed stable.
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);
  context.drawImage(source as CanvasImageSource, 0, 0, width, height);
  return { canvas, width, height };
}

function toBlob(canvas: HTMLCanvasElement, type: string, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode image.'))), type, quality);
  });
}

export async function prepareImage(file: Blob): Promise<Prepared> {
  const source = await decode(file);
  const full = draw(source, EMBED_MAX);
  const thumb = draw(source, THUMB_MAX);

  const embed = await toBlob(full.canvas, 'image/jpeg', 0.9);
  const thumbBlob = await toBlob(thumb.canvas, 'image/webp', 0.82).catch(() => toBlob(thumb.canvas, 'image/jpeg', 0.8));

  if ('close' in source) (source as ImageBitmap).close();

  return {
    thumb: thumbBlob,
    thumbType: thumbBlob.type || 'image/webp',
    embed,
    width: full.width,
    height: full.height,
  };
}

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|avif|bmp|svg)$/i;

export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || IMAGE_EXT.test(file.name);
}

/** Walks a drop (including folders) and returns flat files. */
export async function filesFromDataTransfer(dataTransfer: DataTransfer): Promise<File[]> {
  const items = Array.from(dataTransfer.items ?? []);
  const entries = items
    .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null))
    .filter((entry): entry is FileSystemEntry => Boolean(entry));

  if (!entries.length) return Array.from(dataTransfer.files ?? []);

  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File | null>((resolve) =>
        (entry as FileSystemFileEntry).file(resolve, () => resolve(null)),
      );
      if (file && isImageFile(file)) out.push(file);
      return;
    }
    if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns at most 100 entries per call, so keep reading until empty.
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve) =>
          reader.readEntries(resolve, () => resolve([])),
        );
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  };

  for (const entry of entries) await walk(entry);
  return out;
}
