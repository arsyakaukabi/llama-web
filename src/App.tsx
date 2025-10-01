// EmbeddingGemma — React + Vite + TypeScript (single-file component)
// ---------------------------------------------------------------
// Place this file in your Vite + React + TypeScript project, for example:
//  src/components/EmbeddingGemma.tsx
//
// Setup steps:
// 1) Create project: `npm create vite@latest my-app -- --template react-ts` or `pnpm create vite ...`
// 2) cd my-app && npm install
// 3) Copy the wllama ESM build into `public/wllama/esm/` from `node_modules/@wllama/wllama/esm/`.
//    Required files: `index.js`, `single-thread/wllama.wasm`, `single-thread/wllama.js`,
//    `multi-thread/wllama.wasm`, `multi-thread/wllama.js`, and `multi-thread/wllama.worker.mjs` if you want multi-threading.
// 4) Start dev server: `npm run dev` and open the page where you include the component.
//
// Notes:
// - This component dynamically imports the `index.js` from `/wllama/esm/index.js` at runtime to avoid bundling WASM files.
// - Use quantized GGUF models (Q4/Q5) and/or split shards to run comfortably in-browser.
// - If you host GGUF on Hugging Face, be mindful of large downloads and possible redirects or auth.

import { useCallback, useEffect, useMemo, useRef, useState, type JSX } from 'react';
import { Wllama } from '@wllama/wllama';

import './App.css'

type Progress = { loaded?: number; total?: number };

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/wllama/esm/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm': '/wllama/esm/multi-thread/wllama.wasm',
};

// const DEFAULT_MODEL = 'https://huggingface.co/second-state/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-Q4_K_M.gguf';
const DEFAULT_MODEL = 'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf'

export default function EmbeddingGemma(): JSX.Element {
  const [wllama, setWllama] = useState<any | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState<number | null>(null);
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL);
  const [text, setText] = useState('Saya sedang menguji embedding Gemma untuk demo WASM di browser.');
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // dynamic import of wllama index.js from public path
  const importWllama = useCallback(async () => {
    if (wllama) return wllama;
    setStatus('initializing runtime');
    // dynamic import from public path
    // const mod: any = await import('/wllama/esm/index.js');
    // const Wllama = mod.Wllama ?? mod.default?.Wllama ?? mod;
    const inst = new Wllama(CONFIG_PATHS);
    setWllama(inst);
    setStatus('runtime ready');
    return inst;
  }, [wllama]);

  const onProgress = useCallback((p: Progress) => {
    if (!p || !p.total) return;
    const pct = Math.round(((p.loaded ?? 0) / (p.total ?? 1)) * 100);
    setProgress(pct);
  }, []);

  const loadModelFromUrl = useCallback(
    async (url: string) => {
      try {
        setStatus('loading model from url');
        setProgress(0);
        const inst = await importWllama();
        const start = Date.now();
        await inst.loadModelFromUrl(url, {
          embeddings: true,
          n_ctx: 2048,
          pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
          progressCallback: onProgress,
        });
        const took = Date.now() - start;
        setStatus(`model loaded (took ${took} ms). dim=${inst.getEmbeddingSize ? inst.getEmbeddingSize() : 'unknown'}`);
        setModelLoaded(true);
      } catch (err: any) {
        console.error(err);
        setStatus('failed to load model: ' + (err?.message ?? String(err)));
        setModelLoaded(false);
      } finally {
        setProgress(null);
      }
    },
    [importWllama, onProgress]
  );

  const loadModelFromFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      try {
        setStatus('loading model from files');
        setProgress(0);
        const inst = await importWllama();
        const start = Date.now();
        const blobs = Array.from(files);
        await inst.loadModel(blobs, {
          embeddings: true,
          n_ctx: 2048,
          pooling_type: 'LLAMA_POOLING_TYPE_MEAN',
          progressCallback: onProgress,
        });
        const took = Date.now() - start;
        setStatus(`model loaded from files (took ${took} ms)`);
        setModelLoaded(true);
      } catch (err: any) {
        console.error(err);
        setStatus('failed to load local model: ' + (err?.message ?? String(err)));
        setModelLoaded(false);
      } finally {
        setProgress(null);
      }
    },
    [importWllama, onProgress]
  );

  const createEmbedding = useCallback(
    async (inputText: string) => {
      if (!wllama || !modelLoaded) {
        setStatus('load a model first');
        return;
      }
      if (!inputText) {
        setStatus('type some text to embed');
        return;
      }
      try {
        setStatus('creating embedding');
        const t0 = Date.now();
        const vec: number[] = await wllama.createEmbedding(inputText, { skipBOS: true, skipEOS: true });
        const took = Date.now() - t0;
        setEmbedding(vec);
        setStatus(`embedding created (took ${took} ms). length=${vec.length}`);
      } catch (err: any) {
        console.error(err);
        setStatus('embedding failed: ' + (err?.message ?? String(err)));
      }
    },
    [wllama, modelLoaded]
  );

  const resetRuntime = useCallback(() => {
    if (wllama) {
      try {
        wllama.release?.();
      } catch (e) {
        // ignore
      }
    }
    setWllama(null);
    setModelLoaded(false);
    setStatus('runtime reset');
    setEmbedding(null);
    setProgress(null);
  }, [wllama]);

  useEffect(() => {
    // optional: auto init runtime for warm-up
    return () => {
      // cleanup on unmount
      try { wllama?.release?.(); } catch (e) { }
    };
  }, [wllama]);

  const first128 = useMemo(() => embedding ? embedding.slice(0, 128) : null, [embedding]);

  return (
    <div className="eg-root">
      <style>{`
     
      `}</style>

      <div className="eg-card">
        <div className="eg-header">
          <div>
            <h2 className="eg-title">Embedding Llama</h2>
            <div className="eg-sub">Mohamad Arsya Kaukabi</div>
          </div>
        </div>

        <div className="panel">
          <label>Model URL (GGUF)</label>
          <div className="row" style={{ gap: 12 }}>
            <textarea
              value={modelUrl}
              onChange={(e) => setModelUrl(e.target.value)}
              placeholder="https://.../model.gguf or leave empty to upload"
            />
          </div>

          <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div className="controls" style={{ flex: 1 }}>
              <button className="btn btn-primary" onClick={() => loadModelFromUrl(modelUrl)}>Load from URL</button>

              <label className="file-label">
                Upload .gguf
                <input ref={fileInputRef} type="file" accept=".gguf" multiple onChange={(e) => loadModelFromFiles(e.target.files)} />
              </label>

              <button className="btn btn-muted" onClick={resetRuntime}>Reset runtime</button>
            </div>

            <div className="status" style={{ minWidth: 180, textAlign: 'right' }}>
              {status}{progress !== null ? ` — ${progress}%` : ''}
            </div>
          </div>
        </div>

        <div className="panel">
          <label>Text to embed</label>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} placeholder="Masukkan teks untuk di-embed..." />

          <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
            <button className="btn btn-primary" onClick={() => createEmbedding(text)}>Create embedding</button>
            <button className="btn btn-muted" onClick={() => { setEmbedding(null); setStatus && setStatus('cleared'); }}>Clear</button>
          </div>
        </div>

        <div className="result">
          <h4>Result</h4>
          <div className="meta">{embedding ? `Vector length: ${embedding.length}` : 'No embedding yet.'}</div>
          <pre>{first128 ? JSON.stringify(first128, null, 2) + (embedding && embedding.length > 128 ? '\n\n... (first 128 elements)' : '') : ''}</pre>
        </div>
      </div>
    </div>
  );


}
