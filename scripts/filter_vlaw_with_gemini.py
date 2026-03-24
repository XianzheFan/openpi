"""
Filter VLAW synthetic trajectories using Gemini as a success judge.

For each episode in the VLAW-generated LeRobot dataset, this script:
1. Reads the agentview video and the task description.
2. Uploads the video to Gemini and asks it to judge whether the task was completed.
3. Removes episodes judged as failures and rewrites the dataset.

Usage:
    python scripts/filter_vlaw_with_gemini.py \
        --input_dir data/libero/vlaw_synthetic \
        --output_dir data/libero/vlaw_synthetic_filtered \
        --gemini_model gemini-3-flash-preview
"""

import dataclasses
import json
import logging
import pathlib
import shutil
import time
from typing import Optional

import imageio
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
import tyro


CHUNK_SIZE = 1000
ENV_FPS = 10


class SuccessJudgment(BaseModel):
    reasoning: str
    success: bool
    confidence: float


@dataclasses.dataclass
class Args:
    input_dir: str = "data/libero/vlaw_synthetic"
    output_dir: str = "data/libero/vlaw_synthetic_filtered"
    gemini_model: str = "gemini-3-flash-preview"
    # Sample N evenly-spaced frames from the video to send as images instead of
    # uploading the full video. Set to 0 to upload the full video file.
    num_sample_frames: int = 0
    # Minimum confidence threshold — episodes below this are discarded as ambiguous
    confidence_threshold: float = 0.5
    # Rate-limit delay between Gemini API calls (seconds)
    api_delay: float = 1.0
    # If True, only print judgments without writing the filtered dataset
    dry_run: bool = False


def _wait_for_file_active(client: genai.Client, file_ref) -> None:
    """Block until a Google GenAI uploaded file is processed."""
    file_info = client.files.get(name=file_ref.name)
    while file_info.state.name == "PROCESSING":
        time.sleep(2)
        file_info = client.files.get(name=file_ref.name)
    if file_info.state.name == "FAILED":
        raise ValueError(f"Video processing failed for file: {file_ref.name}")


def _sample_frames_from_video(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Read a video and return N evenly-spaced frames as numpy arrays."""
    reader = imageio.get_reader(video_path, "ffmpeg")
    all_frames = [np.asarray(f) for f in reader]
    reader.close()
    if len(all_frames) == 0:
        return []
    indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
    return [all_frames[i] for i in indices]


def judge_episode_with_gemini(
    client: genai.Client,
    video_path: str,
    task_description: str,
    model: str,
    num_sample_frames: int = 0,
) -> Optional[SuccessJudgment]:
    """Use Gemini to judge whether a trajectory video completes the task.

    Args:
        client: Google GenAI client.
        video_path: Path to the agentview .mp4 video.
        task_description: The language instruction for this task.
        model: Gemini model name.
        num_sample_frames: If >0, send N sampled frames as images instead of uploading the video.

    Returns:
        SuccessJudgment or None on API failure.
    """
    prompt = f"""You are a robot manipulation task success evaluator.

A robot was given this task instruction: "{task_description}"

You are shown a video (or sequence of frames) of the robot's trajectory.

Your job is to determine whether the robot **successfully completed** the task described in the instruction.

Criteria for success:
- The robot must have fully accomplished the goal described in the instruction.
- Partial progress does NOT count as success.
- If the robot is still in the process of executing and has not reached the goal state, that is a failure.
- If the robot knocked over objects, dropped things, or made errors, that is a failure.

Please respond with:
- "reasoning": a brief explanation of what you observe in the video and why you judge it as success or failure
- "success": true if the task was completed, false otherwise
- "confidence": a float between 0.0 and 1.0 indicating how confident you are in your judgment
"""

    try:
        if num_sample_frames > 0:
            # Send sampled frames as images
            frames = _sample_frames_from_video(video_path, num_sample_frames)
            if not frames:
                return None
            contents = [prompt, "\n[Trajectory frames from first to last]:"]
            for i, frame in enumerate(frames):
                # Encode frame as PNG bytes
                import io
                from PIL import Image
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SuccessJudgment,
                    temperature=0.0,
                ),
            )
        else:
            # Upload full video
            video_file = client.files.upload(file=video_path)
            _wait_for_file_active(client, video_file)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        "\n[Robot trajectory video]:",
                        video_file,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SuccessJudgment,
                        temperature=0.0,
                    ),
                )
            finally:
                try:
                    client.files.delete(name=video_file.name)
                except Exception:
                    pass

        result = json.loads(response.text)
        return SuccessJudgment(**result)

    except Exception as e:
        logging.error(f"Gemini API error for {video_path}: {e}")
        return None


def load_dataset_metadata(input_dir: pathlib.Path) -> tuple[list[dict], dict[int, str]]:
    """Load episodes metadata and task descriptions from the dataset.

    Returns:
        (episodes_list, task_index_to_description)
    """
    episodes = []
    episodes_file = input_dir / "meta" / "episodes.jsonl"
    with open(episodes_file) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))

    tasks = {}
    tasks_file = input_dir / "meta" / "tasks.jsonl"
    with open(tasks_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            tasks[entry["task_index"]] = entry["task"]

    return episodes, tasks


def get_video_path(input_dir: pathlib.Path, episode_index: int) -> str:
    """Get the agentview video path for a given episode."""
    chunk_idx = episode_index // CHUNK_SIZE
    return str(
        input_dir
        / "videos"
        / f"chunk-{chunk_idx:03d}"
        / "observation.images.agentview"
        / f"episode_{episode_index:06d}.mp4"
    )


def get_episode_task_index(input_dir: pathlib.Path, episode_index: int) -> int:
    """Read the task_index from the episode's parquet file."""
    chunk_idx = episode_index // CHUNK_SIZE
    pq_path = (
        input_dir
        / "data"
        / f"chunk-{chunk_idx:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    df = pd.read_parquet(pq_path)
    return int(df["task_index"].iloc[0])


def write_filtered_dataset(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    kept_episodes: list[int],
    judgments: dict[int, SuccessJudgment],
) -> None:
    """Copy only the kept episodes to the output directory, re-indexing them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)

    episodes_orig, tasks = load_dataset_metadata(input_dir)
    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)
    shutil.copy2(input_dir / "meta" / "modality.json", output_dir / "meta" / "modality.json")

    # Build new episode list with re-indexed episode_index
    new_episodes = []
    new_global_frame = 0
    stat_states = []
    stat_actions = []

    for new_idx, old_idx in enumerate(tqdm(kept_episodes, desc="Writing filtered episodes")):
        old_chunk = old_idx // CHUNK_SIZE
        new_chunk = new_idx // CHUNK_SIZE

        # Copy and re-index parquet
        old_pq = (
            input_dir / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_idx:06d}.parquet"
        )
        df = pd.read_parquet(old_pq)
        n_frames = len(df)

        df["episode_index"] = new_idx
        df["index"] = range(new_global_frame, new_global_frame + n_frames)
        df["success"] = True  # These are the ones Gemini judged as successful

        new_data_dir = output_dir / "data" / f"chunk-{new_chunk:03d}"
        new_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(new_data_dir / f"episode_{new_idx:06d}.parquet", index=False)

        # Accumulate stats
        for _, row in df.iterrows():
            stat_states.append(np.array(row["observation.state"]))
            stat_actions.append(np.array(row["action"]))

        for cam_key in ["observation.images.agentview", "observation.images.wrist"]:
            old_vid = (
                input_dir / "videos" / f"chunk-{old_chunk:03d}" / cam_key
                / f"episode_{old_idx:06d}.mp4"
            )
            new_vid_dir = output_dir / "videos" / f"chunk-{new_chunk:03d}" / cam_key
            new_vid_dir.mkdir(parents=True, exist_ok=True)
            new_vid_path = new_vid_dir / f"episode_{new_idx:06d}.mp4"
            if old_vid.exists():
                shutil.copy2(old_vid, new_vid_path)

        # Find original episode metadata
        orig_meta = next(
            (e for e in episodes_orig if e["episode_index"] == old_idx), None
        )
        new_episodes.append({
            "episode_index": new_idx,
            "tasks": orig_meta["tasks"] if orig_meta else [],
            "length": n_frames,
            "success": True,
            "synthetic": True,
            "gemini_judgment": dataclasses.asdict(judgments[old_idx])
            if old_idx in judgments
            else None,
        })
        new_global_frame += n_frames

    # Write meta/episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # Write meta/tasks.jsonl (same as original)
    shutil.copy2(input_dir / "meta" / "tasks.jsonl", output_dir / "meta" / "tasks.jsonl")

    # Write meta/info.json
    info["total_episodes"] = len(kept_episodes)
    info["total_frames"] = new_global_frame
    info["filtering_method"] = "gemini_success_judge"
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Write meta/stats.json
    stats = {}
    if stat_states:
        states_arr = np.stack(stat_states)
        stats["observation.state"] = {
            "mean": states_arr.mean(0).tolist(),
            "std": states_arr.std(0).tolist(),
            "min": states_arr.min(0).tolist(),
            "max": states_arr.max(0).tolist(),
            "q01": np.quantile(states_arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(states_arr, 0.99, axis=0).tolist(),
        }
    if stat_actions:
        actions_arr = np.stack(stat_actions)
        stats["action"] = {
            "mean": actions_arr.mean(0).tolist(),
            "std": actions_arr.std(0).tolist(),
            "min": actions_arr.min(0).tolist(),
            "max": actions_arr.max(0).tolist(),
            "q01": np.quantile(actions_arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions_arr, 0.99, axis=0).tolist(),
        }
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logging.info(
        f"Filtered dataset written to {output_dir}: "
        f"{len(kept_episodes)} episodes, {new_global_frame} frames."
    )


def main(args: Args) -> None:
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    episodes, tasks = load_dataset_metadata(input_dir)
    logging.info(f"Loaded {len(episodes)} episodes, {len(tasks)} tasks from {input_dir}")
    client = genai.Client(http_options={"api_version": "v1alpha"})

    judgments: dict[int, SuccessJudgment] = {}
    kept_episodes: list[int] = []

    for ep in tqdm(episodes, desc="Judging episodes with Gemini"):
        ep_idx = ep["episode_index"]
        video_path = get_video_path(input_dir, ep_idx)

        if not pathlib.Path(video_path).exists():
            logging.warning(f"Video not found for episode {ep_idx}: {video_path}")
            continue

        task_idx = get_episode_task_index(input_dir, ep_idx)
        task_desc = tasks.get(task_idx, "unknown task")

        judgment = judge_episode_with_gemini(
            client=client,
            video_path=video_path,
            task_description=task_desc,
            model=args.gemini_model,
            num_sample_frames=args.num_sample_frames,
        )

        if judgment is None:
            logging.warning(f"Episode {ep_idx}: Gemini returned no result, skipping.")
            continue

        judgments[ep_idx] = judgment

        if judgment.success and judgment.confidence >= args.confidence_threshold:
            kept_episodes.append(ep_idx)
            status = "KEPT"
        else:
            status = "REMOVED"

        logging.info(
            f"Episode {ep_idx} [{task_desc}] -> {status} "
            f"(success={judgment.success}, conf={judgment.confidence:.2f}, "
            f"reason={judgment.reasoning[:80]}...)"
        )

        if args.api_delay > 0:
            time.sleep(args.api_delay)

    total = len(episodes)
    kept = len(kept_episodes)
    removed = total - kept
    logging.info(
        f"\nFiltering complete: {kept}/{total} episodes kept, {removed} removed."
    )

    log_path = input_dir / "gemini_judgments.jsonl"
    with open(log_path, "w") as f:
        for ep_idx, j in sorted(judgments.items()):
            f.write(
                json.dumps({
                    "episode_index": ep_idx,
                    "success": j.success,
                    "confidence": j.confidence,
                    "reasoning": j.reasoning,
                })
                + "\n"
            )
    logging.info(f"Judgment log saved to {log_path}")

    if args.dry_run:
        logging.info("Dry run mode — not writing filtered dataset.")
        return

    write_filtered_dataset(input_dir, output_dir, kept_episodes, judgments)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = tyro.cli(Args)
    main(args)
