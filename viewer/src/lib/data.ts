import { useEffect, useState } from "react";
import type { Manifest, Snapshot } from "./types";

const RUNS = "./runs";

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} loading ${url}`);
  return res.json();
}

export interface LoadedRun {
  manifest: Manifest;
  /** Filled in as snapshots arrive; `undefined` = still downloading. */
  snapshots: (Snapshot | undefined)[];
  loaded: number;
}

/**
 * Load a run's manifest, then its snapshots in the background (first one
 * first, so the page can show something right away). Big runs have tens of
 * megabytes of history, so we never make people wait for all of it.
 */
export function useRun(id: string) {
  const [run, setRun] = useState<LoadedRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRun(null);
    setError(null);

    (async () => {
      const manifest = await getJSON<Manifest>(`${RUNS}/${id}/manifest.json`);
      const snapshots: (Snapshot | undefined)[] = manifest.checkpoints.map(() => undefined);
      let loaded = 0;
      const fetchOne = async (i: number) => {
        snapshots[i] = await getJSON<Snapshot>(`${RUNS}/${id}/${manifest.checkpoints[i].file}`);
        loaded++;
        if (!cancelled) setRun({ manifest, snapshots: [...snapshots], loaded });
      };
      await fetchOne(0);
      // The rest, in timeline order, a few at a time.
      let next = 1;
      const workers = Array.from({ length: 4 }, async () => {
        while (!cancelled && next < snapshots.length) await fetchOne(next++);
      });
      await Promise.all(workers);
    })().catch((e) => !cancelled && setError(String(e)));

    return () => {
      cancelled = true;
    };
  }, [id]);

  return { run, error };
}
