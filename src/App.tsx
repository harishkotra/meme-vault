import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Embedder, type WorkerEvent } from './embed/client';
import { DOWNLOAD_MB, type Device } from './embed/config';
import {
  clearVault,
  estimateUsage,
  getOriginal,
  loadMeta,
  putRecords,
  setVector,
  type MemeMeta,
} from './vault/db';
import { filesFromDataTransfer, isImageFile, prepareImage } from './vault/ingest';

type Phase = 'boot' | 'loading' | 'ready' | 'error';
type Hit = { meme: MemeMeta; score: number };

const urlCache = new WeakMap<Blob, string>();
function urlFor(blob: Blob): string {
  let url = urlCache.get(blob);
  if (!url) {
    url = URL.createObjectURL(blob);
    urlCache.set(blob, url);
  }
  return url;
}

const mb = (bytes: number) => `${(bytes / 1_048_576).toFixed(1)} MB`;

export default function App() {
  const embedderRef = useRef<Embedder | null>(null);
  const setUpRef = useRef(false);

  const [phase, setPhase] = useState<Phase>('boot');
  const [device, setDevice] = useState<Device | null>(null);
  const [deviceNote, setDeviceNote] = useState<string | null>(null);
  const [status, setStatus] = useState('Starting the model…');
  const [download, setDownload] = useState({ loaded: 0, total: 0, pct: 0 });
  const [error, setError] = useState<string | null>(null);

  const [memes, setMemes] = useState<MemeMeta[]>([]);
  const [indexing, setIndexing] = useState<{ done: number; total: number } | null>(null);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [usage, setUsage] = useState(0);

  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [lastQuery, setLastQuery] = useState('');
  const [searchSeq, setSearchSeq] = useState(0);
  const [narrow, setNarrow] = useState(false);
  const [open, setOpen] = useState<MemeMeta | null>(null);
  const [online, setOnline] = useState(() => navigator.onLine);
  const [dragging, setDragging] = useState(false);

  const searchRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  const indexed = useMemo(() => memes.filter((meme) => meme.vector).length, [memes]);

  // --- boot: read the vault, then start the model -------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const stored = await loadMeta();
      if (cancelled) return;
      setMemes(stored);
      const { usage } = await estimateUsage();
      setUsage(usage);
      const embedder = new Embedder();
      embedderRef.current = embedder;
      embedder.on((event: WorkerEvent) => {
        if (event.kind === 'download') {
          setDownload({ loaded: event.loaded, total: event.total, pct: event.progress * 100 });
        } else if (event.kind === 'status') {
          setStatus(event.message);
        }
      });
      try {
        setPhase('loading');
        const result = await embedder.load();
        if (cancelled) return;
        setDevice(result.device);
        setDeviceNote(result.reason);
        setPhase('ready');
        setStatus('Ready');
      } catch (caught: any) {
        if (cancelled) return;
        setError(caught?.message ?? String(caught));
        setPhase('error');
      }
    })();
    return () => {
      cancelled = true;
      embedderRef.current?.dispose();
      embedderRef.current = null;
    };
  }, []);

  useEffect(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener('online', up);
    window.addEventListener('offline', down);
    return () => {
      window.removeEventListener('online', up);
      window.removeEventListener('offline', down);
    };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (open) setOpen(null);
        else if (document.activeElement === searchRef.current) searchRef.current?.blur();
      }
      if (event.key === '/' && document.activeElement !== searchRef.current) {
        const tag = (event.target as HTMLElement)?.tagName;
        if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
          event.preventDefault();
          searchRef.current?.focus();
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  // --- ingest ------------------------------------------------------------
  const ingest = useCallback(
    async (files: File[]) => {
      const embedder = embedderRef.current;
      if (!embedder || phase !== 'ready') return;

      const existing = new Set(memes.map((meme) => `${meme.name}:${meme.bytes}`));
      const candidates = files.filter(isImageFile).filter((file) => {
        const key = `${file.name}:${file.size}`;
        if (existing.has(key)) return false;
        existing.add(key);
        return true;
      });

      if (!candidates.length) return;

      const records: MemeMeta[] = [];
      const originals = new Map<string, Blob>();
      const embedBlobs: Blob[] = [];
      const failures: string[] = [];

      setSkipped([]);
      setIndexing({ done: 0, total: candidates.length });

      for (const [position, file] of candidates.entries()) {
        try {
          const prepared = await prepareImage(file);
          const id = `${Date.now().toString(36)}-${position}-${Math.random().toString(36).slice(2, 8)}`;
          const record: MemeMeta = {
            id,
            name: file.name,
            addedAt: Date.now(),
            bytes: file.size,
            width: prepared.width,
            height: prepared.height,
            thumb: prepared.thumb,
            thumbType: prepared.thumbType,
            vector: null,
          };
          records.push(record);
          originals.set(id, file);
          embedBlobs.push(prepared.embed);
        } catch {
          failures.push(file.name);
        }
        if (position % 8 === 7) setIndexing({ done: 0, total: candidates.length });
      }

      if (failures.length) setSkipped(failures);
      if (!records.length) {
        setIndexing(null);
        return;
      }

      await putRecords(records, originals);
      setMemes((current) => [...records, ...current]);

      // Embed in small groups so a tab crash keeps the work already done.
      const groupSize = device === 'wasm' ? 4 : 16;
      for (let start = 0; start < embedBlobs.length; start += groupSize) {
        const slice = embedBlobs.slice(start, start + groupSize);
        const vectors = await embedder.embedImages(slice);
        const ids = records.slice(start, start + slice.length).map((record) => record.id);
        for (const [offset, id] of ids.entries()) {
          const vector = vectors[offset];
          if (!vector) continue;
          await setVector(id, vector);
          setMemes((current) =>
            current.map((meme) => (meme.id === id ? { ...meme, vector } : meme)),
          );
        }
        setIndexing({ done: Math.min(start + slice.length, embedBlobs.length), total: embedBlobs.length });
      }

      setIndexing(null);
      const after = await estimateUsage();
      setUsage(after.usage);
    },
    [device, memes, phase],
  );

  const onDrop = useCallback(
    async (event: React.DragEvent) => {
      event.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      const files = await filesFromDataTransfer(event.dataTransfer);
      await ingest(files);
    },
    [ingest],
  );

  // --- search ------------------------------------------------------------
  const runSearch = useCallback(
    async (text: string) => {
      const embedder = embedderRef.current;
      const pool = memes.filter((meme) => meme.vector);
      if (!embedder || !text.trim() || !pool.length) {
        setHits(null);
        return;
      }
      setSearching(true);
      try {
        const queryVector = await embedder.embedQuery(text);
        const scored = pool.map((meme) => {
          const vector = meme.vector!;
          let dot = 0;
          for (let i = 0; i < vector.length; i++) dot += vector[i] * queryVector[i];
          return { meme, score: dot };
        });
        scored.sort((a, b) => b.score - a.score);
        const top = scored[0]?.score ?? 0;
        setHits(scored.slice(0, 120));
        if (top <= 0) setHits([]);
        setLastQuery(text);
        setSearchSeq((value) => value + 1);
      } catch (caught: any) {
        setError(caught?.message ?? String(caught));
      } finally {
        setSearching(false);
      }
    },
    [memes],
  );

  // Debounced live search: the model already runs locally, so typing is cheap.
  useEffect(() => {
    if (!query.trim()) {
      setHits(null);
      return;
    }
    const timer = setTimeout(() => runSearch(query), 400);
    return () => clearTimeout(timer);
  }, [query, runSearch]);

  const visible = useMemo(() => {
    const list: Hit[] = hits ?? memes.map((meme) => ({ meme, score: 0 }));
    if (hits && narrow && hits.length) {
      const top = hits[0].score;
      return hits.filter((hit) => hit.score >= top * 0.92);
    }
    return list;
  }, [hits, memes, narrow]);

  const totalBytes = useMemo(() => memes.reduce((sum, meme) => sum + meme.bytes, 0), [memes]);

  const reset = async () => {
    if (!confirm('Delete every meme and vector from this browser? The originals on your disk are untouched.')) return;
    await clearVault();
    setMemes([]);
    setHits(null);
    setQuery('');
    setUsage(0);
  };

  const busy = phase === 'loading' || Boolean(indexing);

  return (
    <div
      className={`app${dragging ? ' is-dragging' : ''}`}
      data-phase={phase}
      data-device={device ?? 'none'}
      data-indexed={indexed}
      data-total={memes.length}
      data-query={lastQuery}
      data-searching={searching ? "true" : "false"}
      data-search-seq={searchSeq}
      onDragEnter={(event) => {
        event.preventDefault();
        dragDepth.current += 1;
        if (event.dataTransfer.types.includes('Files')) setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <header className="bar">
        <div className="brand">
          <span className="wordmark">Meme Vault</span>
          <span className="tagline">your memes, found by description, never uploaded</span>
        </div>
        <div className="bar-right">
          <span className={`pill ${online ? '' : 'pill-live'}`}>
            <i className={`dot ${online ? '' : 'dot-live'}`} />
            {online ? 'online' : 'offline, still working'}
          </span>
          {device && <span className="pill pill-quiet">{device === 'webgpu' ? 'WebGPU' : 'WASM'}</span>}
          <span className="pill pill-quiet">{memes.length} memes</span>
          <label className="button button-quiet" tabIndex={0}>
            Add memes
            <input
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                event.target.value = '';
                void ingest(files);
              }}
            />
          </label>
          <label className="button button-quiet" tabIndex={0}>
            Add folder
            <input
              type="file"
              multiple
              hidden
              // @ts-expect-error non-standard but supported in Chromium and Safari
              webkitdirectory=""
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                event.target.value = '';
                void ingest(files);
              }}
            />
          </label>
          {memes.length > 0 && (
            <button type="button" className="button button-quiet" onClick={reset}>
              Clear
            </button>
          )}
        </div>
      </header>

      {(phase === 'loading' || indexing) && (
        <div className="loader">
          <div className="loader-head">
            <span>{indexing ? `Reading your memes ${indexing.done}/${indexing.total}` : status}</span>
            <span className="mono">
              {indexing
                ? `${Math.round((indexing.done / Math.max(1, indexing.total)) * 100)}%`
                : download.total
                  ? `${mb(download.loaded)} / ${mb(download.total)}`
                  : `about ${DOWNLOAD_MB.webgpu} MB`}
            </span>
          </div>
          <div className="track">
            <div
              className="track-fill"
              style={{
                width: indexing
                  ? `${(indexing.done / Math.max(1, indexing.total)) * 100}%`
                  : `${Math.max(2, download.pct)}%`,
              }}
            />
          </div>
          <p className="hint">
            {indexing
              ? 'Embedding runs on this machine. Nothing is being uploaded.'
              : 'One-time model download, cached by your browser. After this, the tab works with the wifi off.'}
          </p>
        </div>
      )}

      {phase === 'error' && (
        <div className="notice notice-error">
          <strong>The model could not start.</strong>
          <p>{error}</p>
          <p className="hint">Reload the page to try again, or try Chrome or Edge 113+ with WebGPU.</p>
        </div>
      )}

      {deviceNote && phase === 'ready' && (
        <div className="notice">
          <strong>Running on WASM.</strong>
          <p>{deviceNote} It works, just slower: one image per batch instead of nine. Chrome with WebGPU loads 284 MB instead of 879 MB.</p>
        </div>
      )}

      <div className={`searchbar${query ? ' has-query' : ''}`}>
        <span className="search-icon" aria-hidden="true">
          ⌕
        </span>
        <input
          ref={searchRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={indexed ? 'the dog in the burning room…' : 'add some memes first'}
          disabled={!indexed}
          aria-label="Describe the meme you are looking for"
        />
        {query && (
          <button type="button" className="clear-query" onClick={() => setQuery('')} aria-label="Clear search">
            ✕
          </button>
        )}
        {hits && (
          <label className="narrow">
            <input type="checkbox" checked={narrow} onChange={(event) => setNarrow(event.target.checked)} />
            close matches only
          </label>
        )}
      </div>

      {searching && <p className="hint hint-inline">found something, checking…</p>}

      {!memes.length && phase === 'ready' && (
        <section className="empty">
          <h1>Drop your meme folder anywhere.</h1>
          <p>
            {indexed === 0 && memes.length === 0
              ? 'PNG, JPG, GIF, WebP. Two thousand at once is fine.'
              : 'Pick the folder your memes actually live in, usually Downloads.'}
          </p>
          <p className="hint">
            Every image is embedded in this tab by a 740M model. Filenames are ignored, so it finds
            things by what they look like. Nothing leaves the machine, and search keeps working with
            the network off.
          </p>
        </section>
      )}

      {hits && hits.length === 0 && (
        <section className="empty">
          <h1>Nothing close.</h1>
          <p>Try describing what is in the image, not the caption you think it had.</p>
        </section>
      )}

      {visible.length > 0 && (
        <section className="grid" aria-live="polite">
          {visible.map(({ meme, score }, position) => (
            <button
              key={meme.id}
              type="button"
              className={`tile${hits && position === 0 ? ' tile-top' : ''}`}
              onClick={() => setOpen(meme)}
              title={meme.name}
            >
              <img src={urlFor(meme.thumb)} alt={meme.name} loading="lazy" decoding="async" />
              {hits && <span className="score mono">{score.toFixed(2)}</span>}
              <span className="tile-name">{meme.name.replace(/\.[^.]+$/, '')}</span>
            </button>
          ))}
        </section>
      )}

      {skipped.length > 0 && (
        <p className="hint hint-inline">
          Skipped {skipped.length} file{skipped.length === 1 ? '' : 's'} the browser could not decode:{' '}
          {skipped.slice(0, 4).join(', ')}
          {skipped.length > 4 ? '…' : ''}
        </p>
      )}

      <footer className="foot">
        <span className="mono">
          {indexed}/{memes.length} indexed · {mb(totalBytes)} of memes · {mb(usage)} site storage
        </span>
        <span className="credit">
          <a href="https://harishkotra.me" target="_blank" rel="noreferrer noopener">
            Built By Harish Kotra
          </a>
          <span className="sep">·</span>
          <a href="https://dailybuild.xyz" target="_blank" rel="noreferrer noopener">
            Checkout my other builds
          </a>
        </span>
        <span className="hint">
          EmbeddingGemma 2 (740M) via Transformers.js · runs on your GPU · press / to search
        </span>
      </footer>

      {dragging && (
        <div className="dropveil">
          <span>Drop it</span>
        </div>
      )}

      {open && <Lightbox meme={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Lightbox({ meme, onClose }: { meme: MemeMeta; onClose: () => void }) {
  const [original, setOriginal] = useState<string | null>(null);

  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    (async () => {
      const blob = await getOriginal(meme.id);
      if (!blob || cancelled) return;
      url = URL.createObjectURL(blob);
      setOriginal(url);
    })();
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [meme.id]);

  return (
    <div className="veil" onClick={onClose} role="dialog" aria-modal="true" aria-label={meme.name}>
      <figure className="figure" onClick={(event) => event.stopPropagation()}>
        <img src={original ?? urlFor(meme.thumb)} alt={meme.name} />
        <figcaption>
          <span className="mono">{meme.name}</span>
          <span className="mono hint">
            {meme.width}×{meme.height} · {mb(meme.bytes)}
          </span>
          <a className="button button-quiet" href={original ?? undefined} download={meme.name} onClick={(e) => e.stopPropagation()}>
            Open original
          </a>
          <button type="button" className="button button-quiet" onClick={onClose}>
            Close
          </button>
        </figcaption>
      </figure>
    </div>
  );
}
