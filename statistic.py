"""
Shapiro-Wilk Test (Normality Check):
This is being used to test whether the NRS scores in each group are normally distributed. If the p-value is less than 0.05, it indicates the data doesn't follow a normal distribution.
3. ANOVA:
ANOVA is performed to see if there are significant differences in NRS scores across the groups. If the p-value is less than 0.05, it suggests that the mean NRS scores differ significantly between the groups.
4. Spearman Correlation:
Correlating PI with NRS Scores: Since the data is non-normally distributed, Spearman's correlation is used. You are checking if there's a positive or negative correlation between PI and NRS scores for each group.
Correlating PI with Pain Duration: Similarly, you check the correlation between PI and the mean pain duration.
"""
#IMPORT DATA

#import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Creazione dei DataFrame per ogni fase
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

#Dati NRS
nrs_scores = {
    "Group 1 (IP 1.4)": [45, 42, 47, 48, 45, 43, 42, 49, 50, 43],
    "Group 2 (IP 1.5)": [47, 48, 45, 49, 50, 54, 47, 59, 60, 43],
    "Group 3 (IP 1.6)": [59, 54, 50, 60, 58, 57, 65, 63],
    "Group 4 (IP 1.7)": [52, 54, 48, 59, 57, 62, 64, 58, 50, 49, 60, 55],
    "Group 5 (IP 1.8)": [70, 68, 66, 73, 69, 72, 71, 67, 70, 72],
    "Group 6 (IP 1.9)": [76, 80, 82, 81, 75, 77, 79, 84],
    "Group 7 (IP 2.0)": [82, 87, 91, 79, 92, 85],
}

#Dati sulla durata del dolore
pain_duration = pd.DataFrame({
    "PI": [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "Min_Hours": [5, 6, 8, 7, 12, 24, 24],
    "Max_Hours": [12, 12, 24, 24, 24, 48, 80],
    "Mean_Duration": [8.5, 9, 16, 15.5, 18, 36, 52]  # Durata media per gruppo
})

#1.ANALISI STATISTICA!!!!!!!!!!!!!!
#Distribuzione dei valori di PI (Intercritical, Frequent Crisis, Critical Phase)
#1.grafico delle tre distribuzioni di PI nelle 3 fasi
#descrizione statistica media devstd
#test shapiro-wilk per verificare se è una distribuzione normale



#2.ANALISI STATISTICA!!!!!!!!!!!!!!!!!
# Confronto tra le tre fasi (Intercritical, Frequent Crisis, Critical Phase)




#3.ANALISI STATISTICA!!!!!!!!!!
# test di normalità su NRS
#Confronto tra NRS e PI (correlazione dolore e resistenza cerebrovascolare)
nrs_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in nrs_scores.items()]))

#passo 1: Test di normalità (Shapiro-Wilk) su NRS
shapiro_results = {col: stats.shapiro(nrs_df[col].dropna()) for col in nrs_df.columns if len(nrs_df[col].dropna()) >= 3}
print("Test di Shapiro-Wilk per la normalità dei punteggi NRS:")
for group, result in shapiro_results.items():
    w_stat, p_val = result
    print(f"{group}: W = {w_stat:.3f}, p-value = {p_val:.3f}")
    if p_val < 0.05:
        print(f" → I dati di {group} non seguono una distribuzione normale (p < 0.05)\n")
    else:
        print(f" → I dati di {group} seguono una distribuzione normale (p >= 0.05)\n")
#sì distribuzione normale
#passo 2: ANOVA per verificare differenze nei gruppi NRS (questo non capisco se serve o meno)
anova_nrs = stats.f_oneway(*[nrs_df[col].dropna() for col in nrs_df.columns])

print(f"ANOVA tra i gruppi NRS: F = {anova_nrs.statistic:.3f}, p-value = {anova_nrs.pvalue:.3f}")
if anova_nrs.pvalue < 0.05:

    print(" → Differenza significativa tra i gruppi NRS (p < 0.05)\n")
else:
    print(" → Nessuna differenza significativa tra i gruppi NRS (p ≥ 0.05)\n")

#Sì c'è una differenza significativa tra i gruppi (quali?)

#passo 3: Correlazione tra PI e NRS (PEARSON, perché i dati sono distribuiti normalmente)
#assicurati che i gruppi abbiano lo stesso numero di dati
for col in nrs_df.columns:
    # Limitare pi_values in modo che abbia la stessa lunghezza dei dati
    if len(nrs_df[col].dropna()) == len([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]):
        correlation, p_value = stats.spearmanr(nrs_df[col].dropna(), [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        print(f"Correlazione Spearman per {col}: {correlation:.3f}, p-value: {p_value:.3f}")



#4.ANALISI STATISTICA!!!!!!!!!!
#Confronto della durata del dolore tra diversi PI

##t-test
# Fase intercritica vs fase critica
t_test_1 = stats.ttest_ind(intercritical_phase["PI"], critical_phase["PI"])
print(f"T-test: Fase Intercritica vs Fase Critica → t = {t_test_1.statistic:.3f}, p-value = {t_test_1.pvalue:.3f}")

# Fase intercritica vs crisi frequenti
t_test_2 = stats.ttest_ind(intercritical_phase["PI"], frequent_crisis_phase["PI"])
print(f"T-test: Fase Intercritica vs Crisi Frequenti → t = {t_test_2.statistic:.3f}, p-value = {t_test_2.pvalue:.3f}")

# Crisi frequenti vs fase critica
t_test_3 = stats.ttest_ind(frequent_crisis_phase["PI"], critical_phase["PI"])
print(f"T-test: Crisi Frequenti vs Fase Critica → t = {t_test_3.statistic:.3f}, p-value = {t_test_3.pvalue:.3f}")

# Creiamo un DataFrame unificato per la correlazione tra PI e NRS
nrs_melted = nrs_df.melt(var_name="Group", value_name="NRS").dropna()
nrs_melted["PI"] = nrs_melted["Group"].str.extract(r"(\d\.\d)").astype(float)

mean_nrs = nrs_df.mean()
pi_values = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # PI per ogni gruppo
pearson_corr, p_value = stats.pearsonr(pi_values, mean_nrs)
print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {p_value:.3f}")

# Stampare il risultato
#spearman_corr = stats.spearmanr(pi_values, mean_nrs)
#print(f"Spearman Correlation: {spearman_corr.correlation:.3f}, p-value: {spearman_corr.pvalue:.3f}")
#Se Spearman Correlation > 0.5 → Correlazione positiva significativa
#Se p-value < 0.05 → Correlazione statisticamente significativa
# Calcolo della correlazione Spearman tra PI e durata media del dolore
spearman_corr, p_value = stats.spearmanr(pain_duration["PI"], pain_duration["Mean_Duration"])

# Stampare il risultato
print(f"\nSpearman Correlation between PI and Pain Duration: {spearman_corr:.3f}, p-value: {p_value:.3f}")
#Se la correlazione Spearman è alta (> 0.5) → All'aumentare del PI, aumenta anche la durata media del mal di testa.
#Se il p-value < 0.05 → La correlazione è statisticamente significativa.

# t-test tra le fasi
# Confronto tra Fase Intercritica e Fase Critica
t_test_result = stats.ttest_ind(intercritical_phase["PI"], critical_phase["PI"], equal_var=False)

# Stampare il risultato del t-test
print(f"\nT-test tra Fase Intercritica e Fase Critica: p-value = {t_test_result.pvalue:.3f}")


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



"""
1. Shapiro-Wilk Test:

Test di normalità per verificare se i dati dell'NRS seguono una distribuzione normale.
Output: Risultati del test per ogni gruppo. Se il p-value è inferiore a 0.05, i dati non sono normali.

2. ANOVA:

Verifica se ci sono differenze significative nei punteggi NRS tra i diversi gruppi.
Output: Risultato dell'ANOVA con il p-value. Se è inferiore a 0.05, c'è una differenza significativa tra i gruppi.

3. Correlazione Spearman (PI vs NRS):

Esamina la correlazione tra l'Indice di Pulsatilità (PI) e l'intensità del dolore (NRS).
Output: Correlazione Spearman tra PI e NRS, con p-value. Se il p-value è inferiore a 0.05, la correlazione è significativa.

4. Durata del Dolore vs PI:

Correlazione tra PI e durata media del dolore.
Output: Correlazione Spearman tra PI e la durata media del dolore, con

"""