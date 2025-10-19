#!/usr/bin/env python3
"""
Evolution run writer: persists per-generation artifacts, diffs, and summaries.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from datetime import datetime
import difflib


class EvolutionRunWriter:
    """
    File-based writer to visualize evolution across generations.
    Creates a run directory with per-generation subdirectories, saving:
    - Candidate code files
    - Candidate metadata (JSON)
    - Unified diffs against parent when available
    - Generation summaries and overall run summary
    """

    def __init__(self, agent_id: str, base_dir: Optional[Path] = None) -> None:
        self.agent_id = agent_id
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "experiments"
        self.run_dir: Optional[Path] = None
        self._code_index: Dict[str, str] = {}

    def start_run(self, task_spec: Dict[str, Any], planning_results: Optional[Dict[str, Any]] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"ea_run_{self.agent_id}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)

        # Save task spec and planning outputs
        with open(self.run_dir / "task_spec.json", "w", encoding="utf-8") as f:
            json.dump(task_spec or {}, f, indent=2)

        if planning_results is not None:
            with open(self.run_dir / "planning_results.json", "w", encoding="utf-8") as f:
                json.dump(planning_results, f, indent=2)

    def write_generation(
        self,
        generation_index: int,
        candidates: List[Any],
        best_candidate: Optional[Any] = None,
        diversity_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.run_dir is None:
            raise RuntimeError("Run directory not initialized. Call start_run() first.")

        gen_dir = self.run_dir / f"gen_{generation_index:03d}"
        os.makedirs(gen_dir, exist_ok=True)

        # Save each candidate's code and metadata
        for cand in candidates:
            cid = getattr(cand, "id", f"cand_{generation_index}")
            code = getattr(cand, "code", "") or ""

            # Write code file
            with open(gen_dir / f"candidate_{cid}.py", "w", encoding="utf-8") as f:
                f.write(code)

            # Metadata JSON
            metadata = {
                "id": cid,
                "generation": getattr(cand, "generation", generation_index),
                "fitness_score": getattr(cand, "fitness_score", 0.0),
                "parent_id": getattr(cand, "parent_id", None),
                "mutation_type": getattr(cand, "mutation_type", None),
                "prompt": getattr(cand, "prompt", ""),
                "tools": getattr(cand, "tools", {}),
                "memory": getattr(cand, "memory", {}),
                "metadata": getattr(cand, "metadata", {}),
            }
            with open(gen_dir / f"candidate_{cid}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Diff against parent if available in index
            parent_id = metadata.get("parent_id")
            if parent_id and parent_id in self._code_index:
                parent_code = self._code_index[parent_id].split("\n")
                child_code = code.split("\n")
                diff = difflib.unified_diff(
                    parent_code,
                    child_code,
                    fromfile=f"parent_{parent_id}",
                    tofile=f"child_{cid}",
                    lineterm=""
                )
                diff_text = "\n".join(diff)
                with open(gen_dir / f"diff_{cid}_from_{parent_id}.patch", "w", encoding="utf-8") as f:
                    f.write(diff_text)

        # Generation summary
        summary = {
            "generation": generation_index,
            "num_candidates": len(candidates),
            "best_candidate_id": getattr(best_candidate, "id", None) if best_candidate else None,
            "best_fitness": getattr(best_candidate, "fitness_score", None) if best_candidate else None,
            "diversity_metrics": diversity_metrics or {},
        }
        with open(gen_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Update index for next generation diffs
        for cand in candidates:
            cid = getattr(cand, "id", None)
            code = getattr(cand, "code", None)
            if cid is not None and code is not None:
                self._code_index[cid] = code

    def end_run(self, final_best: Optional[Any] = None, evolution_stats: Optional[Dict[str, Any]] = None) -> None:
        if self.run_dir is None:
            return
        summary = {
            "final_best_candidate_id": getattr(final_best, "id", None) if final_best else None,
            "final_best_fitness": getattr(final_best, "fitness_score", None) if final_best else None,
            "evolution_stats": evolution_stats or {},
            "completed_at": datetime.now().isoformat(),
        }
        with open(self.run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


