"""Export and integrate the trained Double Agent Champion into submission.py."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKPOINT_PATH = ROOT / "evoexperiments" / "checkpoints" / "best_double_agent.json"

def export():
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint {CHECKPOINT_PATH} does not exist.")
        return

    with open(CHECKPOINT_PATH, "r") as f:
        champ = json.load(f)

    genome = champ["genome"]
    gen = champ.get("generation", 0)
    money = champ.get("best_money", 0)
    fit = champ.get("best_fitness", 0)

    print("=" * 65)
    print(f"EXPORTING DOUBLE AGENT CHAMPION (Generation {gen})")
    print(f"Benchmark Money: ${money:,.0f} | Fitness: {fit:,.0f}")
    print("Genome settings:")
    for k, v in genome.items():
        print(f"  {k}: {v}")
    print("=" * 65)

    # Read submission.py
    sub_file = ROOT / "submission.py"
    with open(sub_file, "r") as f:
        sub_code = f.read()

    # Read engine code
    engine_file = ROOT / "evoexperiments" / "double_agent_engine.py"
    with open(engine_file, "r") as f:
        eng_code = f.read()

    formatted_genome = (
        "CHAMPION_GENOME: Dict[str, Any] = "
        + json.dumps(genome, indent=4)
        .replace("true", "True")
        .replace("false", "False")
        .replace("null", "None")
    )

    # In engine, get from GENOME_SPECS or get_town_drain down to end
    drain_idx = eng_code.find("def get_town_drain")
    engine_body = eng_code[drain_idx:].replace("DEFAULT_GENOME", "CHAMPION_GENOME")

    champ_idx = sub_code.find("CHAMPION_GENOME: Dict[str, Any] = {")
    if champ_idx != -1:
        base_sub = sub_code[:champ_idx]
    else:
        base_sub = sub_code

    entrypoint = """

# ------------------------------------------------------------
# Competition entry point
# ------------------------------------------------------------

_agent = DoubleAgentPolicy(CHAMPION_GENOME)


def agent(obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _agent(obs, *args, **kwargs)
"""
    final_submission_code = base_sub + formatted_genome + "\n\n" + engine_body + entrypoint

    with open(sub_file, "w") as f:
        f.write(final_submission_code)
    print(f"Successfully compiled champion into {sub_file}")

    main_file = ROOT / "main.py"
    with open(main_file, "w") as f:
        f.write(final_submission_code)
    print(f"Successfully compiled champion into {main_file}")

    import tarfile
    tar_path = ROOT / "submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(main_file, arcname="main.py")
        tar.add(sub_file, arcname="submission.py")
    print(f"Successfully created submission package: {tar_path} ({tar_path.stat().st_size} bytes)")

if __name__ == "__main__":
    export()
