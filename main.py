import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Creazione dei DataFrame per le diverse fasi ---
intercritical_phase = pd.DataFrame({
    "PI": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "n_pazienti": [31, 39, 21, 13, 5, 3],
    "PIC": [4, 5, 6, 7, 8, 9]
})

frequent_crisis_phase = pd.DataFrame({
    "PI": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "n_pazienti": [2, 6, 12, 7, 6, 5],
    "PIC": [5, 6, 7, 8, 9, 10]
})

critical_phase = pd.DataFrame({
    "PI": [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "n_pazienti": [10, 15, 8, 12, 10, 9, 6],
    "PIC": [14, 15, 16, 17, 18, 19, 20]
})

# --- Test di Normalità per tutte le variabili ---
def test_normality(data, name):
    shapiro_test = stats.shapiro(data)
    print(f"Test di normalità (Shapiro-Wilk) per {name}: p-value = {shapiro_test.pvalue:.3f}")
    return shapiro_test.pvalue

# --- Analisi Statistica: Test di Normalità per le fasi ---
p_intercritical = test_normality(intercritical_phase["PI"], "Intercritical Phase")
p_frequent = test_normality(frequent_crisis_phase["PI"], "Frequent Crisis Phase")
p_critical = test_normality(critical_phase["PI"], "Critical Phase")

# --- Test ANOVA (se i dati sono normali) o Kruskal-Wallis ---
if all(p > 0.05 for p in [p_intercritical, p_frequent, p_critical]):
    anova_result = stats.f_oneway(intercritical_phase["PI"], frequent_crisis_phase["PI"], critical_phase["PI"])
    print("Se p-value>0.05 --> i dati seguono una distribuzione normale\nEseguo test ANOVA\n")
    print(f"ANOVA test: F = {anova_result.statistic:.3f}, p-value = {anova_result.pvalue:.3f}")
    #SE c'è una differenza tra gruppi eseguo post-hoc test

    # Test post-hoc Tukey se ANOVA è significativa
    if anova_result.pvalue < 0.05:
        print("L'ANOVA mostra che c'è una differenza statisticamente significativa tra i gruppi (p < 0.05).\nQuesto indica che almeno una delle tre fasi ha un valore medio di PI significativamente diverso dalle altre.\nEseguo Post-hoc test\n")
        all_data = pd.concat([
            intercritical_phase.assign(group="Intercritical"),
            frequent_crisis_phase.assign(group="Frequent"),
            critical_phase.assign(group="Critical")
        ])
        tukey_result = pairwise_tukeyhsd(all_data["PI"], all_data["group"])
        print(tukey_result)
else:
    kruskal_result = stats.kruskal(intercritical_phase["PI"], frequent_crisis_phase["PI"], critical_phase["PI"])
    print("p-value >0.05 --> i dati NON seguono una distribuzione normale\nEseguo test KRUSKAL\n")
    print(f"Kruskal-Wallis test: H = {kruskal_result.statistic:.3f}, p-value = {kruskal_result.pvalue:.3f}")

# --- Grafico della distribuzione di PI nelle tre fasi ---
plt.figure(figsize=(10, 5))
sns.histplot(intercritical_phase["PI"], kde=True, color="blue", label="Intercritical Phase", bins=6)
sns.histplot(frequent_crisis_phase["PI"], kde=True, color="green", label="Frequent Crisis Phase", bins=6)
sns.histplot(critical_phase["PI"], kde=True, color="red", label="Critical Phase", bins=7)
plt.xlabel("Pulsatility Index (PI)")
plt.ylabel("Number of Patients")
plt.title("Distribution of PI in Different Phases")
plt.legend()
plt.show()

# --- Analisi della correlazione tra PI e NRS ---
nrs_scores = {
    "Group 1 (PI 1.4)": [45, 42, 47, 48, 45, 43, 42, 49, 50, 43],
    "Group 2 (PI 1.5)": [47, 48, 45, 49, 50, 54, 47, 59, 60, 43],
    "Group 3 (PI 1.6)": [59, 54, 50, 60, 58, 57, 65, 63],
    "Group 4 (PI 1.7)": [52, 54, 48, 59, 57, 62, 64, 58, 50, 49, 60, 55],
    "Group 5 (PI 1.8)": [70, 68, 66, 73, 69, 72, 71, 67, 70, 72],
    "Group 6 (PI 1.9)": [76, 80, 82, 81, 75, 77, 79, 84],
    "Group 7 (PI 2.0)": [82, 87, 91, 79, 92, 85]
}
nrs_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in nrs_scores.items()]))
shapiro_results = {col: stats.shapiro(nrs_df[col].dropna()) for col in nrs_df.columns if len(nrs_df[col].dropna()) >= 3}
print("Test di Shapiro-Wilk per la normalità dei punteggi NRS:")
for group, result in shapiro_results.items():
    w_stat, p_val = result
    print(f"{group}: W = {w_stat:.3f}, p-value = {p_val:.3f}")
    if p_val < 0.05:
        print(f" → I dati di {group} non seguono una distribuzione normale (p < 0.05)\n")

        # Test di ANOVA per confrontare le medie di NRS nei gruppi di PI
        p_nrs = test_normality(nrs_df.melt()["value"].dropna(), "NRS")
        if p_nrs > 0.05:
            anova_nrs = stats.f_oneway(*[nrs_df[col].dropna() for col in nrs_df.columns])
            print(f"ANOVA su NRS tra i gruppi: F = {anova_nrs.statistic:.3f}, p-value = {anova_nrs.pvalue:.3f}")
            if anova_nrs.pvalue < 0.05:
                tukey_nrs = pairwise_tukeyhsd(nrs_df.melt()["value"], nrs_df.melt()["variable"])
                print(tukey_nrs)
        else:
            print("I dati di NRS non sono normali. Considerare test non parametrico post-hoc.")
    else:
        print(f" → I dati di {group} seguono una distribuzione normale (p >= 0.05)\n")

# Calcolo della correlazione Spearman tra PI e NRS
pi_values = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
mean_nrs = nrs_df.mean()
pearson_corr = stats.pearsonr(pi_values, mean_nrs)
print(f"Pearson Correlation tra PI e NRS: {pearson_corr[0]:.3f}, p-value: {pearson_corr[1]:.3f}")

# Grafico correlazione PI vs NRS
plt.scatter(pi_values, mean_nrs, color="blue")
plt.xlabel("Pulsatility Index (PI)")
plt.ylabel("Mean NRS Score")
plt.title("Correlation between PI and Pain Intensity (NRS)")
plt.show()

# --- Analisi della durata del dolore ---
pain_duration = pd.DataFrame({
    "PI": [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "Mean_Duration": [8.5, 9, 16, 15.5, 18, 36, 52]
})

# Correlazione Spearman tra PI e durata del dolore
pearson_pain = stats.pearsonr(pain_duration["PI"], pain_duration["Mean_Duration"])
print(f"Pearson Correlation tra PI e durata del dolore: {pearson_pain[0]:.3f}, p-value: {pearson_pain[1]:.3f}")

"""
# Correlazione Spearman tra PI e durata del dolore
spearman_pain, p_value_pain = stats.spearmanr(pain_duration["PI"], pain_duration["Mean_Duration"])
print(f"Spearman Correlation tra PI e durata del dolore: {spearman_pain:.3f}, p-value: {p_value_pain:.3f}")
"""

# Grafico correlazione PI vs durata dolore
plt.scatter(pain_duration["PI"], pain_duration["Mean_Duration"], color="red")
plt.xlabel("Pulsatility Index (PI)")
plt.ylabel("Mean Pain Duration (Hours)")
plt.title("Correlation between PI and Pain Duration")
plt.show()

# #######################GRAFICI#################
#1. Visualizzazione delle distribuzioni PI nelle tre fasi
plt.figure(figsize=(10, 5))
sns.barplot(x=intercritical_phase["PI"], y=intercritical_phase["n_pazienti"], color="blue", label="Intercritical Phase")
sns.barplot(x=frequent_crisis_phase["PI"], y=frequent_crisis_phase["n_pazienti"], color="green", label="Frequent Crisis Phase")
sns.barplot(x=critical_phase["PI"], y=critical_phase["n_pazienti"], color="red", label="Critical Phase")
plt.xlabel("Pulsatility Index (PI)")
plt.ylabel("Number of Patients")
plt.title("Distribution of PI in Different Phases")
plt.legend()
plt.show()

#2. Visualizzare il grafico della correlazione PI e NRS
plt.scatter(pi_values, mean_nrs, color="blue")
plt.xlabel("Pulsatility Index (PI)")
plt.ylabel("Mean NRS Score")
plt.title("Correlation between PI and Pain Intensity (NRS)")
plt.show()

#3. Rifacciamo il plot della distribuzione di NRS per ogni gruppo
plt.figure(figsize=(10, 5))
sns.boxplot(data=nrs_df, notch=True)
plt.xlabel("PI Group")
plt.ylabel("NRS Score")
plt.title("Distribution of NRS Scores by PI Group")
plt.show()

#4 Creare il grafico della durata del dolore in funzione del PI
plt.figure(figsize=(8,5))
plt.scatter(pain_duration["PI"], pain_duration["Mean_Duration"], color="red", label="Mean Pain Duration")
plt.plot(pain_duration["PI"], pain_duration["Mean_Duration"], linestyle="--", color="black", alpha=0.6)
plt.xlabel("Pulsatility Index (PI)")
plt.ylabel("Mean Pain Duration (Hours)")
plt.title("Correlation between PI and Pain Duration")
plt.legend()
plt.show()

