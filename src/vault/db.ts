/**
 * The vault lives in IndexedDB: a small `meta` store (thumbnail + vector) for the
 * grid and search, and a `files` store for the original bytes. Splitting them keeps
 * startup fast even with thousands of memes.
 */
export type MemeMeta = {
  id: string;
  name: string;
  addedAt: number;
  bytes: number;
  width: number;
  height: number;
  thumb: Blob;
  thumbType: string;
  vector: Float32Array | null;
};

const DB_NAME = 'meme-vault';
const DB_VERSION = 1;
const META = 'meta';
const FILES = 'files';

let dbPromise: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(META)) {
        const store = db.createObjectStore(META, { keyPath: 'id' });
        store.createIndex('addedAt', 'addedAt');
      }
      if (!db.objectStoreNames.contains(FILES)) db.createObjectStore(FILES);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open IndexedDB.'));
  });
  return dbPromise;
}

function tx(db: IDBDatabase, stores: string[], mode: IDBTransactionMode) {
  return db.transaction(stores, mode);
}

function done(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB write failed.'));
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB write aborted.'));
  });
}

export async function loadMeta(): Promise<MemeMeta[]> {
  const db = await open();
  const store = tx(db, [META], 'readonly').objectStore(META);
  const rows = await new Promise<MemeMeta[]>((resolve, reject) => {
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result as MemeMeta[]);
    request.onerror = () => reject(request.error);
  });
  return rows.sort((a, b) => b.addedAt - a.addedAt);
}

export async function putRecords(records: MemeMeta[], originals: Map<string, Blob>): Promise<void> {
  if (!records.length) return;
  const db = await open();
  const transaction = tx(db, [META, FILES], 'readwrite');
  const meta = transaction.objectStore(META);
  const files = transaction.objectStore(FILES);
  for (const record of records) meta.put(record);
  for (const [id, blob] of originals) files.put(blob, id);
  await done(transaction);
}

export async function setVector(id: string, vector: Float32Array): Promise<void> {
  const db = await open();
  const transaction = tx(db, [META], 'readwrite');
  const store = transaction.objectStore(META);
  const request = store.get(id);
  request.onsuccess = () => {
    const record = request.result as MemeMeta | undefined;
    if (record) {
      record.vector = vector;
      store.put(record);
    }
  };
  await done(transaction);
}

export async function getOriginal(id: string): Promise<Blob | null> {
  const db = await open();
  const store = tx(db, [FILES], 'readonly').objectStore(FILES);
  return new Promise((resolve, reject) => {
    const request = store.get(id);
    request.onsuccess = () => resolve((request.result as Blob | undefined) ?? null);
    request.onerror = () => reject(request.error);
  });
}

export async function clearVault(): Promise<void> {
  const db = await open();
  const transaction = tx(db, [META, FILES], 'readwrite');
  transaction.objectStore(META).clear();
  transaction.objectStore(FILES).clear();
  await done(transaction);
}

export async function estimateUsage(): Promise<{ usage: number; quota: number }> {
  if (!navigator.storage?.estimate) return { usage: 0, quota: 0 };
  const { usage = 0, quota = 0 } = await navigator.storage.estimate();
  return { usage, quota };
}
