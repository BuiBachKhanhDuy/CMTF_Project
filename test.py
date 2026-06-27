from src.benchmark.ablation_config import generate_grid
cfgs = generate_grid("data_ablation")
print(len(cfgs))
for c in cfgs:
    print(c.cell_id)