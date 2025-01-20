# %%0
import pandas as pd
import seaborn as sns

from mlsae.eval import output_dir

df = pd.read_csv(output_dir / "evaluation_results.csv")
df["Architecture"] = [""] * len(df)
df.loc[df["encoder_dim"].notna() & df["decoder_dim"].notna(), "Architecture"] = "1 Hidden Enc + Dec"
df.loc[df["encoder_dim"].notna() & df["decoder_dim"].isna(), "Architecture"] = "1 Hidden Enc"
df.loc[df["encoder_dim"].isna() & df["decoder_dim"].isna(), "Architecture"] = "Shallow SAE"
print(df)

# %%
plot = sns.lineplot(data=df, x="avg_nonzero_acts", y="avg_mse", hue="Architecture", marker="o")
plot.set_xlabel("$L_0$")
plot.set_ylabel("MSE")
# add more y ticks
plot.set_xscale("log")
# plot.set_yscale("log")
plot.grid(True)
# make gridlines dashed
plot.grid(linestyle='--')
plot.legend(title="")
# Save as svg
plot.get_figure().savefig(output_dir / "mse_vs_l0.svg", format="svg")



