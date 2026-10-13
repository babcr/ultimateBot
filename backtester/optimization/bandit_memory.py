"""Explicit, model-scoped administration of the common learning database."""
import argparse
import json
from pathlib import Path

from ultimateBot.backtester.optimization.bandit_store import LearningStore
from ultimateBot.backtester.optimization.contextual_bandit import DEFAULT_MEMORY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--model-id")
    export = commands.add_parser("export")
    export.add_argument("--model-id")
    export.add_argument("--output", type=Path, required=True)
    load = commands.add_parser("import")
    load.add_argument("--input", type=Path, required=True)
    reset = commands.add_parser("reset")
    reset.add_argument("--model-id", required=True)
    args = parser.parse_args()
    store = LearningStore(args.memory, read_only=args.command in ("inspect", "export"))
    if args.command == "inspect":
        data = store.export(args.model_id)
        models = [{"model_id": row["model_id"], "identity": json.loads(row["identity_json"]),
                   "decision": json.loads(row["state_json"])["decision"], "memory": store.summary(row["model_id"])}
                  for row in data["tables"]["models"]]
        print(json.dumps(models, indent=2, ensure_ascii=False))
    elif args.command == "export":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(store.export(args.model_id), indent=2, ensure_ascii=False), encoding="utf-8")
        print(args.output)
    elif args.command == "import":
        store.import_memory(json.loads(args.input.read_text(encoding="utf-8")))
        print("Imported without resetting any existing model")
    else:
        store.reset_model(args.model_id)
        print(f"Reset model {args.model_id}")


if __name__ == "__main__":
    main()
