"""One-shot repair: rebuild leg kernels' train cell from the leg1 template."""
import json
import pathlib

leg1 = json.loads(pathlib.Path("kaggle/kernel-k6-leg1/MyAI_100M_Leg1.ipynb")
                  .read_text(encoding="utf-8"))
tmpl = "".join(leg1["cells"][3]["source"])

for leg, stop in ((2, "10500"), (3, "12750"), (4, "15000")):
    src = (tmpl.replace("stop-step 7500", f"stop-step {stop}")
               .replace("ckpts-leg1", f"ckpts-leg{leg}")
               .replace("train-leg1", f"train-leg{leg}")
               .replace("LEG1_OK", f"LEG{leg}_OK"))
    nb = json.loads(pathlib.Path(f"kaggle/kernel-k6-leg{leg}/MyAI_100M_Leg{leg}.ipynb")
                    .read_text(encoding="utf-8"))
    nb["cells"][3]["source"] = src.splitlines(keepends=True)
    # summary cell path fixes (leg1 lines copied earlier may still say leg1)
    nb["cells"][4]["source"] = [
        f"!test -f /kaggle/working/ckpts-leg{leg}/final.pt && echo LEG{leg}_OK\n",
        ("!python -c \"import json; r=json.load(open('/kaggle/working/train-leg"
         f"{leg}.json')); [print(k, r.get(k)) for k in ('steps','first_loss',"
         "'final_train_loss','val_loss','tokens_seen')]\"")]
    pathlib.Path(f"kaggle/kernel-k6-leg{leg}/MyAI_100M_Leg{leg}.ipynb").write_text(
        json.dumps(nb, indent=1), encoding="utf-8")
    print(f"leg{leg} rebuilt, stop={stop}")

# sanity: every leg train cell parses with exactly one stop-step + leg-correct outdir
for leg, stop in ((1, "7500"), (2, "10500"), (3, "12750"), (4, "15000")):
    nb = json.loads(pathlib.Path(f"kaggle/kernel-k6-leg{leg}/MyAI_100M_Leg{leg}.ipynb")
                    .read_text(encoding="utf-8"))
    src = "".join(nb["cells"][3]["source"])
    assert src.count("--stop-step") == 1 and f"stop-step {stop}" in src, f"leg{leg} stop"
    assert src.count("ckpts-leg") == 1 and f"ckpts-leg{leg}" in src, f"leg{leg} outdir"
    assert "python scripts/train.py" in src, f"leg{leg} cmd missing"
    assert src.count("--resume") == 1, f"leg{leg} resume"
print("all 4 legs verified")
