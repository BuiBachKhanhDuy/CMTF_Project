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
    """Run a single prediction (auto-routed), with optional step-by-step trace."""
    from .config import MultiAgentConfig
    from .graph import run_graph
    from .trace import build_manifest, write_trace_file, render_step

    trace_file = getattr(args, "trace_file", None)
    config = MultiAgentConfig(
        evaluation_mode=getattr(args, "eval", False),
        trace_enabled=bool(getattr(args, "trace", False) or trace_file),
    )
    query_text = args.query or f"Should I buy {args.symbol} for {args.horizon} day?"
    result = run_graph(
        query_text=query_text,
        cutoff=args.cutoff,
        horizon=args.horizon,
        symbol=args.symbol,
        config=config,
    )

    records = result.get("trace", [])
    manifest = build_manifest(config, eval_mode=config.evaluation_mode,
                              seed=getattr(args, "seed", None),
                              extra={"symbol": args.symbol, "cutoff": args.cutoff,
                                     "horizon": args.horizon})
    final_answer = result.get("answer_text") or ""

    if getattr(args, "trace", False) and records:
        print("\n=== RUN TRACE " + "=" * 50)
        for i, rec in enumerate(records, 1):
            print(render_step(rec, i, len(records)))
        print("=" * 64 + "\n")

    if trace_file:
        write_trace_file(trace_file, manifest, records, final_answer)
        logger.info("Trace transcript → {}", trace_file)

    if getattr(args, "json", None):
        from pathlib import Path
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"manifest": manifest, "state": json.loads(_state_to_json(result))},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Machine record → {}", args.json)

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
                "gate_pred": result.get("gate_pred"),
                "final_pred": result.get("final_pred"),
                "baseline_pred": result.get("baseline_pred"),
                "news_residual": result.get("news_residual"),
                "gate_tau": result.get("gate_tau"),
                "gate_coverage": result.get("gate_coverage"),
                "risk_vetoed": result.get("risk_vetoed"),
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


def cmd_rank(args):
    """Cross-sectional ranking branch — rank N symbols for one date (matched scope)."""
    from .config import MultiAgentConfig
    from .agents.rank_agent import rank_agent_node

    config = MultiAgentConfig()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    state = {
        "target_symbols": symbols,
        "target_horizon_days": args.horizon,
        "prediction_time": args.cutoff,
        "node_timings": {},
    }
    result = rank_agent_node(state, config)
    print(json.dumps({
        "date": args.cutoff, "horizon": args.horizon,
        "longs": result["rank_longs"], "shorts": result["rank_shorts"],
        "abstained": result["rank_abstained"],
        "ranking": result["ranking"], "warnings": result["warnings"],
    }, indent=2, ensure_ascii=False, default=_json_serializable))


def cmd_research(args):
    """Research branch — grounded news RAG for a symbol (no trade call)."""
    from .config import MultiAgentConfig
    from .agents.research_agent import research_agent_node
    from src.pipeline.orchestrator import prepare_single_cutoff

    config = MultiAgentConfig(evaluation_mode=getattr(args, "eval", False))
    data = prepare_single_cutoff(
        symbol=args.symbol, cutoff=args.cutoff, sequence_len=config.sequence_len,
        news_cache_dir=str(config.news_cache_dir),
        sentiment_output_dir=str(config.sentiment_output_dir),
    )
    state = {"articles": data.get("articles", []), "prediction_time": args.cutoff,
             "aspect_filter": "general", "node_timings": {}}
    result = research_agent_node(state, config)
    print(json.dumps({
        "summary": result["research_summary_vi"],
        "retrieved_docs": result["retrieved_docs"],
    }, indent=2, ensure_ascii=False, default=_json_serializable))


def cmd_eval(args):
    """Run the agent-ablation evaluation ladder (A0-A5) + §10.8 decision rule."""
    from .eval_ladder import run_ladder

    result = run_ladder(horizon=args.horizon, n_boot=args.n_boot)
    print(json.dumps({
        "calibration": result["calibration"],
        "cross_sectional": result["cross_sectional"],
        "decision": result["decision"],
        "out_dir": result["out_dir"],
    }, indent=2, ensure_ascii=False, default=_json_serializable))
    logger.info("✓ Eval ladder written → {}", result["out_dir"])


def cmd_h3(args):
    """H3 experiment: MAS vs a plain LLM call.

    mode=forecaster (default, the decisive test): an INDEPENDENT LLM forecaster that
      sees only raw prices+news+technicals vs the gated MAS, on directional accuracy
      and calibration (AURC).
    mode=faithfulness: does the grounded+critic MAS hallucinate fewer numbers than a
      bare LLM given the same fact sheet.
    """
    if args.mode == "forecaster":
        from .h3_faithfulness import run_forecaster_h3
        summary = run_forecaster_h3(horizon=args.horizon, n=args.n, seed=args.seed)
    else:
        from .h3_faithfulness import run_faithfulness
        summary = run_faithfulness(horizon=args.horizon, n=args.n, seed=args.seed)
    printable = {k: v for k, v in summary.items() if k != "records"}
    print(json.dumps(printable, indent=2, ensure_ascii=False, default=_json_serializable))
    logger.info("✓ H3 ({}) → {}", args.mode, summary["out_path"])


def cmd_metalabel_eval(args):
    """Metalabel-agent eval: MAS baseline vs MAS+metalabel vs plain LLM, same 280-row sample."""
    from .metalabel_eval import run_metalabel_eval
    summary = run_metalabel_eval(horizon=args.horizon)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_serializable))
    logger.info("✓ metalabel-eval → {}", summary["out_path"])


def cmd_improved_eval(args):
    """Improved (de-biased, subset-selected) MAS vs LLM, leak-free time-split."""
    from .improved_ensemble import compare_mas_vs_llm, build_improved_mas
    imp = build_improved_mas(horizon=args.horizon)
    from .improved_ensemble import _battery
    ev = imp["eval_mask"]
    print(json.dumps({
        "selected_subset": imp["selected_subset"], "cut_date": imp["cut_date"],
        "improved_MAS_eval": _battery(imp["ens"][ev], imp["truth"][ev]),
        "champion_lstm_eval": _battery(imp["champion"][ev], imp["truth"][ev]),
    }, indent=2, default=_json_serializable))
    print("\n=== H1/H2 BATTERY — multi-CMTF ensemble MAS vs single champion (leak-free) ===")
    from .improved_ensemble import h1_h2_battery
    print(json.dumps(h1_h2_battery(horizon=args.horizon), indent=2, ensure_ascii=False, default=_json_serializable))
    print("\n=== IMPROVED MAS vs PLAIN LLM (shared eval rows) ===")
    print(json.dumps(compare_mas_vs_llm(horizon=args.horizon), indent=2, ensure_ascii=False, default=_json_serializable))


def cmd_calibrate(args):
    """Freeze the validation-calibrated GatePolicy artifact (VN_{H}d.json).

    Reads the cached VALIDATION predictions of the pre-registered CMTF_CORE cell
    (never TEST — leak-free) and writes the frozen policy the gate_agent loads.
    """
    from .config import MultiAgentConfig
    from .gate_io import calibrate_from_cache

    config = MultiAgentConfig()
    policy, meta, out_path = calibrate_from_cache(
        pred_dir="cache/predictions",
        gate_dir=config.gate_policy_dir,
        horizon=args.horizon,
        coverage=config.gate_coverage,
        gate_on_raw_seed=config.gate_on_raw_seed,
        seed=config.ensemble_seeds[0],
        cmtf_version=config.cmtf_version,
        backbone_version=config.backbone_version,
        conviction=config.use_conviction_sizing,
    )
    logger.info(
        "✓ Calibrated GatePolicy {}d → {} | tau={:.5f} coverage={:.3f} conviction_scale={:.5f} "
        "val_score={:.4f} (n_val={}, seed={}, raw_seed={})",
        args.horizon, out_path, policy.tau, policy.coverage, policy.conviction_scale,
        policy.val_score, meta["n_val"], meta["calibration_seed"], meta["gate_on_raw_seed"],
    )
    print(json.dumps({"path": str(out_path), "policy": {
        "tau": policy.tau, "conviction": policy.conviction,
        "conviction_scale": policy.conviction_scale, "coverage": policy.coverage,
        "val_score": policy.val_score}, **meta}, indent=2, ensure_ascii=False))


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
    p_predict.add_argument("--trace", action="store_true",
                           help="Print a step-by-step trace of every node")
    p_predict.add_argument("--trace-file", dest="trace_file",
                           help="Write a human-readable Markdown transcript to this path")
    p_predict.add_argument("--json", help="Write a machine-readable run record to this path")
    p_predict.add_argument("--seed", type=int, help="Seed recorded in the run manifest")
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

    # rank subcommand (comparison branch)
    p_rank = subparsers.add_parser("rank", help="Cross-sectional ranking of N symbols (matched scope)")
    p_rank.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. VCB,CTG,BID")
    p_rank.add_argument("--horizon", type=int, required=True, choices=[1, 5, 20])
    p_rank.add_argument("--cutoff", required=True, help="Prediction date (YYYY-MM-DD)")
    p_rank.set_defaults(func=cmd_rank)

    # research subcommand (RAG branch)
    p_research = subparsers.add_parser("research", help="Grounded news RAG for a symbol (no trade call)")
    p_research.add_argument("--symbol", required=True, help="Stock ticker")
    p_research.add_argument("--cutoff", required=True, help="As-of date (YYYY-MM-DD)")
    p_research.add_argument("--eval", action="store_true", help="LLM-free deterministic digest")
    p_research.set_defaults(func=cmd_research)

    # eval subcommand (agent-ablation ladder + decision rule)
    p_eval = subparsers.add_parser("eval", help="Run the A0-A5 ladder + §10.8 decision rule")
    p_eval.add_argument("--horizon", type=int, default=5, choices=[1, 5, 20])
    p_eval.add_argument("--n-boot", dest="n_boot", type=int, default=5000,
                        help="Paired-bootstrap resamples for ΔAURC CI")
    p_eval.set_defaults(func=cmd_eval)

    # h3 subcommand (MAS vs a plain LLM call)
    p_h3 = subparsers.add_parser("h3", help="H3: MAS vs a plain LLM call (forecaster | faithfulness)")
    p_h3.add_argument("--mode", choices=["forecaster", "faithfulness"], default="forecaster",
                      help="forecaster = independent LLM vs gated MAS (decisive); "
                           "faithfulness = grounded+critic vs bare LLM on the same facts")
    p_h3.add_argument("--horizon", type=int, default=5, choices=[1, 5, 20])
    p_h3.add_argument("--n", type=int, default=56, help="Sample size (stratified across symbols)")
    p_h3.add_argument("--seed", type=int, default=0)
    p_h3.set_defaults(func=cmd_h3)

    # improved-eval subcommand (improved MAS vs LLM)
    p_ie = subparsers.add_parser("improved-eval", help="Improved de-biased+subset MAS vs LLM (leak-free)")
    p_ie.add_argument("--horizon", type=int, default=5, choices=[1, 5, 20])
    p_ie.set_defaults(func=cmd_improved_eval)

    # metalabel-eval subcommand (metalabel agent vs MAS baseline vs plain LLM)
    p_me = subparsers.add_parser("metalabel-eval",
                                 help="MAS baseline vs MAS+metalabel vs plain LLM (same 280-row sample)")
    p_me.add_argument("--horizon", type=int, default=5, choices=[1, 5, 20])
    p_me.set_defaults(func=cmd_metalabel_eval)

    # calibrate subcommand
    p_calib = subparsers.add_parser(
        "calibrate", help="Freeze the validation-calibrated GatePolicy (VN_{H}d.json)")
    p_calib.add_argument("--horizon", type=int, required=True, choices=[1, 5, 20],
                         help="Forecast horizon in days")
    p_calib.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    args.func(args)


if __name__ == "__main__":
    main()
