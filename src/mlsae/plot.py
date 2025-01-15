# %%0
import pandas as pd
import seaborn as sns

from mlsae.eval import output_dir

df = pd.read_csv(output_dir / "evaluation_results.csv")

# %%
plot = sns.barplot(data=df, x="architecture_name", y="avg_mse", hue="k")
plot.bar_label(plot.containers[0], fmt="%.2e")
plot.bar_label(plot.containers[1], fmt="%.2e")
plot.bar_label(plot.containers[2], fmt="%.2e")
