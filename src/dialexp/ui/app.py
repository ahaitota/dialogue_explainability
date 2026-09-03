"""Local inspection UI for every stage of the pipeline.

    uv run streamlit run src/dialexp/ui/app.py

Read-only: it inspects `results/`, it never launches a run. Pick a task, setup and
example once in the sidebar and every tab shows that same example, so one case can
be followed across Step A, the four B experiments and Step C.
"""
from __future__ import annotations

import json

import streamlit as st

from dialexp.ui import charts, data

st.set_page_config(page_title="Dialogue explainability", layout="wide")

config = data.get_config()

# ---- sidebar: the one example every tab follows --------------------------
st.sidebar.title("Example")
task = st.sidebar.selectbox("Task", config.tasks)
setup = st.sidebar.selectbox("Setup", config.setups)
ids = data.example_ids(config, task, setup)
if not ids:
    st.sidebar.error("No Step A results for this combination.")
    row_id = None
else:
    row_id = st.sidebar.selectbox("Example id", ids)
st.sidebar.caption(f"Model: `{data.model_name(config)}`")

step_a_rows = data.load_rows(str(data.stage_path(config, "results/step_a", task, setup)))
step_a = data.row_by_id(step_a_rows, row_id)


def missing(stage: str, command: str) -> None:
    st.info(f"**{stage}** has no results for `{task}` / `{setup}` yet.")
    st.code(command, language="bash")


def show_messages(messages: list[dict]) -> None:
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content")
        if content is None:
            continue
        with st.chat_message("assistant" if role in ("assistant", "tool") else "user"):
            st.caption(role)
            if role == "tool":
                try:
                    st.json(json.loads(content), expanded=False)
                except json.JSONDecodeError:
                    st.text(content)
            else:
                st.markdown(content)


def answer_badges(row: dict) -> None:
    left, mid, right = st.columns(3)
    left.metric("parsed answer", str(row.get("parsed_answer")))
    mid.metric("target", str(row.get("target"))[:40])
    right.metric("finish reason", str(row.get("finish_reason")))


tabs = st.tabs([
    "Overview", "Step A", "Ask-why", "B1 AttnLRP", "B2 Patching",
    "B3 Context masking", "B4 Logic masking",
    "C1 Grounded explanation", "C2 LLM judge", "Cross-stage",
])

# ---- Overview ------------------------------------------------------------
with tabs[0]:
    st.header("What has been run")
    st.dataframe(data.inventory(config), width="stretch", hide_index=True)
    st.caption("Rows per (stage, task, setup). Zero means that stage has not produced results yet.")

# ---- Step A --------------------------------------------------------------
with tabs[1]:
    st.header("Step A — generation")
    if step_a is None:
        missing("Step A", f"uv run python scripts/run_step_a.py configs/experiment.yaml --task {task}")
    else:
        answer_badges(step_a)
        show_messages(step_a["messages"])
        if step_a.get("cot"):
            with st.expander("Reasoning (`cot`)"):
                st.text(step_a["cot"])
        st.subheader("Response")
        st.markdown(step_a.get("response") or "_empty_")
        if step_a.get("tool_calls"):
            st.subheader("Tool calls")
            st.json(step_a["tool_calls"])

# ---- Ask-why -------------------------------------------------------------
with tabs[2]:
    st.header("Ask-why baseline")
    why = data.row_by_id(data.load_rows(str(data.stage_path(config, "results/ask_why", task, setup))), row_id)
    if why is None:
        missing("Ask-why", "uv run python scripts/run_ask_why.py configs/experiment.yaml")
    else:
        st.caption(
            "The **whole** Step A conversation is replayed — system prompt, user turns and tool "
            "results — then the model's own answer is appended as an assistant turn, then the "
            "why-question. The saved reasoning (`cot`) is deliberately **not** replayed, so the "
            "model must re-derive a justification.",
        )
        left, right = st.columns(2)
        left.subheader("Answer being explained")
        left.markdown(why.get("source_response") or "_empty_")
        right.subheader("Self-reported explanation")
        right.markdown(why.get("explanation") or "_empty_")
        if why.get("explanation_cot"):
            with st.expander("Explanation reasoning"):
                st.text(why["explanation_cot"])
        if step_a is not None:
            with st.expander("Exact context the model was given"):
                show_messages(step_a["messages"])
                with st.chat_message("assistant"):
                    st.caption("assistant (its own answer, replayed)")
                    st.markdown(why.get("source_response") or "_empty_")
                with st.chat_message("user"):
                    st.caption("user")
                    st.markdown(why.get("why_prompt") or "")

# ---- B1 ------------------------------------------------------------------
with tabs[3]:
    st.header("B1 — AttnLRP attention saliency")
    variant = st.radio("Variant", list(data.B1_VARIANTS), horizontal=True)
    b1_rows = data.load_rows(str(data.stage_path(config, data.B1_VARIANTS[variant], task, setup)))
    if not b1_rows:
        missing("B1", "uv run python scripts/run_b1.py configs/experiment.yaml")
    else:
        st.subheader("Relevance split per example")
        st.pyplot(charts.b1_region_figure(b1_rows))
        b1 = data.row_by_id(b1_rows, row_id)
        if b1 is None:
            st.warning(f"Example {row_id} was skipped in this variant (see the run log for the reason).")
        else:
            cols = st.columns(4)
            cols[0].metric("prompt", f"{(b1.get('mean_prompt_relevance') or 0):.1%}")
            cols[1].metric("reasoning", f"{(b1.get('mean_reasoning_relevance') or 0):.1%}")
            cols[2].metric("answer-so-far", f"{(b1.get('mean_answer_relevance') or 0):.1%}")
            cols[3].metric("explained tokens", b1.get("n_explained"))
            st.subheader("Token-level heatmap")
            png = charts.b1_token_heatmap_png(b1, variant)
            if png:
                st.image(png, width="stretch")
                st.caption(f"Variant: **{variant}**. `[VALUE→]`/`[←VALUE]` bracket the located "
                           "answer span. Token 0 is excluded from the colour scale (attention sink).")
            else:
                st.warning("This row has no `tokens`/`token_relevance` — re-run B1 to backfill.")

# ---- B2 ------------------------------------------------------------------
with tabs[4]:
    st.header("B2 — Activation patching")
    b2_rows = data.load_rows(str(data.stage_path(config, config.b2["results_dir"], task, setup)))
    if not b2_rows:
        missing("B2", "sbatch --export=ALL,STAGE=b2,CONFIG=configs/experiment.yaml scripts/run_aic.sh")
    else:
        b2 = data.row_by_id(b2_rows, row_id)
        if b2 is None:
            st.warning(f"Example {row_id} was skipped by B2 (see the job log for the reason).")
        else:
            cols = st.columns(4)
            cols[0].metric("corrupted", f"{b2['corrupted_from']} → {b2['corrupted_to']}")
            cols[1].metric("occurrences swapped", b2.get("corrupted_occurrences", "?"))
            cols[2].metric("read at token", b2["fact_start"])
            cols[3].metric("restated in", b2.get("restated_in", "?"))
            st.caption(f"Field `{b2['corrupted_field']}` · "
                       f"written token {b2['clean_token']!r} vs rival {b2['corrupt_token']!r} · "
                       f"LD clean {b2['logit_diff_clean']}, corrupt {b2['logit_diff_corrupt']}")
            st.subheader("Patching effect by layer")
            st.pyplot(charts.b2_layer_figure(b2))
            st.caption("1.0 = fully restored (denoising) or fully destroyed (noising); 0 = no effect.")
            st.subheader("False-positive check (Heimersheim & Nanda §4.2)")
            region = st.selectbox("Region", sorted(b2["region_sizes"]))
            st.pyplot(charts.b2_logit_figure(b2, region))
            st.caption("A genuine restoration raises the written token's logit. If only the rival's "
                       "logit falls, the patch merely damaged the model.")
            with st.expander("Region sizes (tokens patched)"):
                st.json(b2["region_sizes"])

# ---- B3 ------------------------------------------------------------------
with tabs[5]:
    st.header("B3 — Context masking")
    b3_rows = data.load_masked(config, config.masks["context_results_dir"], task, setup)
    if not b3_rows:
        missing("B3", "uv run python scripts/run_b3.py configs/experiment.yaml")
    else:
        st.pyplot(charts.b3_change_figure(b3_rows))
        for row in [r for r in b3_rows if r.get("id") == row_id]:
            verdict = ("not found in the user turns" if not row.get("found")
                       else "answer CHANGED" if row.get("answer_changed")
                       else "answer unchanged" if row.get("answer_changed") is False
                       else "no verdict")
            with st.expander(f"`{row.get('masked_field')}` = {row.get('masked_value')!r} — {verdict}"):
                left, right = st.columns(2)
                left.caption(f"Step A answer: {row.get('ref_parsed_answer')}")
                left.markdown((step_a or {}).get("response") or "_n/a_")
                right.caption(f"After masking: {row.get('parsed_answer')}")
                right.markdown(row.get("response") or "_not rerun_")

# ---- B4 ------------------------------------------------------------------
with tabs[6]:
    st.header("B4 — Logic masking")
    b4_rows = data.load_masked(config, config.masks["logic_results_dir"], task, setup)
    if not b4_rows:
        missing("B4", "uv run python scripts/run_b4.py configs/experiment.yaml")
        st.caption("B4 needs tools to intercept, so it exists only for `dialogue`, "
                   "and only for tasks with entries in `configs/masks/logic_masks.yaml`.")
    else:
        st.dataframe(
            [{"id": r["id"], "tool": r["mask"]["tool"], "mode": r["mask"]["mode"],
              "tool called": r.get("masked_tool_called"), "answer changed": r.get("answer_changed"),
              "parsed": r.get("parsed_answer"), "reference": r.get("ref_parsed_answer")}
             for r in b4_rows],
            width="stretch", hide_index=True,
        )
        for row in [r for r in b4_rows if r.get("id") == row_id]:
            mask = row["mask"]
            state = ("tool never called — inconclusive" if not row.get("masked_tool_called")
                     else "answer CHANGED" if row.get("answer_changed") else "answer unchanged")
            with st.expander(f"`{mask['tool']}` ({mask['mode']}) — {state}"):
                left, right = st.columns(2)
                left.caption(f"Step A answer: {row.get('ref_parsed_answer')}")
                left.markdown((step_a or {}).get("response") or "_n/a_")
                right.caption(f"With the tool masked: {row.get('parsed_answer')}")
                right.markdown(row.get("response") or "_not rerun_")

# ---- C1: grounded explanation -------------------------------------------
with tabs[7]:
    st.header("C1 — Grounded explanation")
    evidence_rows = data.load_rows(str(data.stage_path(config, config.step_c["evidence_dir"], task, setup)))
    if not evidence_rows:
        missing("Step C evidence", "uv run python scripts/run_step_c.py configs/experiment.yaml --phase evidence")
    else:
        evidence = data.row_by_id(evidence_rows, row_id)
        if evidence is None:
            st.warning(f"Example {row_id} lacked a required B stage and was dropped.")
        else:
            from dialexp.evidence import render_evidence

            st.subheader("Verified causes handed to the synthesizer")
            st.caption(f"Sources: {', '.join(evidence.get('sources', []))} · "
                       "only B3/B4 verdicts are causal; B1/B2 appear as SUPPORTING and may not "
                       "be cited as reasons.")
            st.text(render_evidence(evidence))

            grounded = data.row_by_id(
                data.load_rows(str(data.stage_path(config, config.step_c["explanations_dir"], task, setup))),
                row_id)
            st.subheader("Generated explanation")
            if grounded is None:
                missing("C1 synthesis",
                        "uv run python scripts/run_step_c.py configs/experiment.yaml --phase synthesis")
            else:
                st.markdown(grounded.get("explanation") or "_empty_")
                if grounded.get("explanation_cot"):
                    with st.expander("Model reasoning"):
                        st.text(grounded["explanation_cot"])
                with st.expander("Exact prompt sent to the model"):
                    st.text(grounded.get("prompt") or "")
            with st.expander("Raw evidence record"):
                st.json(evidence.get("evidence", {}))

# ---- C2: LLM judge -------------------------------------------------------
with tabs[8]:
    st.header("C2 — LLM judge")
    judged = data.row_by_id(
        data.load_rows(str(data.stage_path(config, config.step_c["judgements_dir"], task, setup))), row_id)
    grounded = data.row_by_id(
        data.load_rows(str(data.stage_path(config, config.step_c["explanations_dir"], task, setup))), row_id)
    why = data.row_by_id(
        data.load_rows(str(data.stage_path(config, "results/ask_why", task, setup))), row_id)

    st.subheader("The two explanations being compared")
    left, right = st.columns(2)
    left.markdown("**Grounded (C1)** — built from verified causes")
    left.markdown((grounded or {}).get("explanation") or "_not generated_")
    right.markdown("**Ask-why baseline** — the model's self-report")
    right.markdown((why or {}).get("explanation") or "_not generated_")

    st.subheader("Scores")
    if judged is None:
        missing("C2 judging", "uv run python scripts/run_step_c.py configs/experiment.yaml --phase judge")
    else:
        rows = [{"arm": arm, **{c: scores[c] for c in judged["scores"][arm]}}
                for arm, scores in judged["scores"].items()]
        st.dataframe(rows, width="stretch", hide_index=True)
        deltas = {c: judged["scores"]["grounded"][c] - judged["scores"]["ask_why"][c]
                  for c in judged["scores"]["grounded"]}
        cols = st.columns(len(deltas))
        for col, (criterion, delta) in zip(cols, deltas.items()):
            col.metric(criterion, judged["scores"]["grounded"][criterion], delta=round(delta, 2),
                       help="grounded minus ask-why")
        st.caption(
            f"Shown to the judge blind and in randomised order as {judged['slots']} · "
            f"in the distractor subset: **{judged['has_distractor']}** "
            "(B proved at least one factor NOT causal, so a self-report has something to get wrong).",
        )
        with st.expander("Exact input the judge saw (identical for the human judge)"):
            st.text(judged.get("judge_input") or "")

# ---- Cross-stage ---------------------------------------------------------
with tabs[9]:
    st.header(f"Example {row_id} across every stage")
    if step_a is None:
        st.info("Pick an example with Step A results.")
    else:
        b1_rows = data.load_rows(str(data.stage_path(
            config, data.B1_VARIANTS["answer_value (value-targeted)"], task, setup)))
        b2_rows = data.load_rows(str(data.stage_path(config, config.b2["results_dir"], task, setup)))
        b3_rows = [r for r in data.load_masked(config, config.masks["context_results_dir"], task, setup)
                   if r.get("id") == row_id]
        b4_rows = [r for r in data.load_masked(config, config.masks["logic_results_dir"], task, setup)
                   if r.get("id") == row_id]
        summary = [
            {"stage": "Step A", "present": True,
             "detail": f"answer {step_a.get('parsed_answer')} vs target {str(step_a.get('target'))[:30]}"},
            {"stage": "Ask-why", "present": bool(data.row_by_id(
                data.load_rows(str(data.stage_path(config, "results/ask_why", task, setup))), row_id)),
             "detail": "self-reported explanation"},
            {"stage": "B1", "present": data.row_by_id(b1_rows, row_id) is not None,
             "detail": "relevance split + token heatmap"},
            {"stage": "B2", "present": data.row_by_id(b2_rows, row_id) is not None,
             "detail": "layer-wise causal trace"},
            {"stage": "B3", "present": bool(b3_rows),
             "detail": f"{sum(1 for r in b3_rows if r.get('answer_changed'))}/{len(b3_rows)} masks changed the answer"},
            {"stage": "B4", "present": bool(b4_rows),
             "detail": f"{sum(1 for r in b4_rows if r.get('answer_changed'))}/{len(b4_rows)} masks changed the answer"},
        ]
        st.dataframe(summary, width="stretch", hide_index=True)
        st.caption("A missing stage means that example was skipped there — the run log records why.")
