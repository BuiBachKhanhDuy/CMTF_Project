"""CLI for the multi-agent inference system."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from loguru import logger


def _json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _state_to_json(state: dict) -> str:
    """Convert a MultiAgentState to a JSON-serializable dict."""
    output = {}
    skip_keys = {"close_window", "market_window", "market_tabular", "news_emb"}
    for k, v in state.items():
        if k in skip_keys:
            continue
        if isinstance(v, np.ndarray):
            if v.size <= 30:
                output[k] = v.tolist()
            else:
                output[k] = f"<ndarray shape={v.shape}>"
        else:
            output[k] = v

    return json.dumps(output, indent=2, default=_json_serializable, ensure_ascii=False)


def cmd_predict(args):
    """Run a single prediction."""
    from .config import MultiAgentConfig
    from .graph import run_graph

    config = MultiAgentConfig(evaluation_mode=getattr(args, 'eval', False))
    query_text = args.query or f"Should I buy {args.symbol} for {args.horizon} day?"
    result = run_graph(
        query_text=query_text,
        cutoff=args.cutoff,
        horizon=args.horizon,
        symbol=args.symbol,
        config=config,
    )

    print(_state_to_json(result))


def cmd_batch_predict(args):
    """Run predictions for a date range."""
    import pandas as pd
    from .config import MultiAgentConfig
    from .graph import run_graph

    config = MultiAgentConfig(evaluation_mode=getattr(args, 'eval', False))

    date_range = pd.bdate_range(start=args.start, end=args.end)

    results = []
    for cutoff_ts in date_range:
        cutoff_str = cutoff_ts.strftime("%Y-%m-%d")
        try:
            query_text = f"Should I buy {args.symbol} for {args.horizon} day?"
            result = run_graph(
                query_text=query_text,
                cutoff=cutoff_str,
                horizon=args.horizon,
                symbol=args.symbol,
                config=config,
            )
            results.append({
                "symbol": args.symbol,
                "cutoff": cutoff_str,
                "horizon": args.horizon,
                "action": result.get("action"),
                "position_scale": result.get("position_scale"),
                "final_pred": result.get("final_pred"),
                "baseline_pred": result.get("baseline_pred"),
                "news_residual": result.get("news_residual"),
                "final_confidence": result.get("final_confidence"),
                "fusion_score": result.get("fusion_decision", {}).get("score"),
                "policy_version": result.get("policy_version", 1),
                "decision_reasoning": result.get("decision_reasoning"),
            })
            logger.info("✓ {} {} → {}", cutoff_str, args.symbol, result.get("action"))
        except Exception as e:
            logger.error("✗ {} {} — {}", cutoff_str, args.symbol, e)
            results.append({
                "symbol": args.symbol,
                "cutoff": cutoff_str,
                "horizon": args.horizon,
                "action": "error",
                "error": str(e),
            })

    df = pd.DataFrame(results)
    output_path = args.output or f"results/multiagent_{args.symbol}_{args.horizon}d.csv"
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Batch results saved → {} ({} rows)", output_path, len(df))
    print(df.to_string(index=False))


def cmd_reflect(args):
    """Settle batch predictions and update policy from realized outcomes."""
    from .batch_reflection import batch_reflection
    from .config import MultiAgentConfig

    config = MultiAgentConfig()
    try:
        result = batch_reflection(
            batch_csv=args.batch,
            symbol=args.symbol,
            config=config,
            min_samples=args.min_samples,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("✓ Reflection complete: v{} → v{}", result["old_version"], result["new_version"])
    except Exception as e:
        logger.error("✗ Reflection failed: {}", e)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Financial Prediction System",
        prog="python -m src.multiagent",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict subcommand
    p_predict = subparsers.add_parser("predict", help="Run a single prediction")
    p_predict.add_argument("--query", help="Natural language query (optional)")
    p_predict.add_argument("--symbol", required=True, help="Stock ticker (e.g. VCB, BID)")
    p_predict.add_argument("--cutoff", required=True, help="Prediction date (YYYY-MM-DD)")
    p_predict.add_argument("--horizon", type=int, required=True, choices=[1, 5, 20],
                           help="Forecast horizon in days")
    p_predict.add_argument("--eval", action="store_true",
                           help="Evaluation mode: disable all LLM calls")
    p_predict.set_defaults(func=cmd_predict)

    # batch-predict subcommand
    p_batch = subparsers.add_parser("batch-predict", help="Run predictions for a date range")
    p_batch.add_argument("--symbol", required=True, help="Stock ticker")
    p_batch.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_batch.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_batch.add_argument("--horizon", type=int, required=True, choices=[1, 5, 20])
    p_batch.add_argument("--output", help="Output CSV path")
    p_batch.add_argument("--eval", action="store_true",
                         help="Evaluation mode: disable all LLM calls")
    p_batch.set_defaults(func=cmd_batch_predict)

    # reflect subcommand
    p_reflect = subparsers.add_parser("reflect", help="Settle batch predictions and update policy")
    p_reflect.add_argument("--batch", required=True, help="Batch predictions CSV file")
    p_reflect.add_argument("--symbol", required=True, help="Stock ticker")
    p_reflect.add_argument("--min-samples", type=int, default=30,
                           help="Minimum settled trades before policy update (default: 30)")
    p_reflect.set_defaults(func=cmd_reflect)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    args.func(args)


if __name__ == "__main__":
    main()
