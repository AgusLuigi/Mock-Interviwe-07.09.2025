# Clustering-App mit Dropdown-Menüs und erweiterter Ellbogen-Einstellung
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, List, Union
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
from ipywidgets import VBox, Dropdown, Button, Output, Label, HTML, SelectMultiple, IntText 

warnings.filterwarnings("ignore") 

# WICHTIGE DEFINITION: df_ML (der vom benutzer ausgewehlte Aktuelle Dataframe zur ML)
df_ML = pd.DataFrame() 

class ClusteringApp:
    def __init__(self, original_df: pd.DataFrame, comparison_column: Union[str, List[str]] = None, treatment_map: Dict[str, str] = None, 
                 elbow_style: str = 'bx-', n_init: Any = 'auto', cat_cleaning_map: Dict[str, str] = None):
        self.original_df = original_df.copy() 
        self.df = original_df.copy() 
        if comparison_column is None:
             self.comparison_col = []
        elif isinstance(comparison_column, str):
             self.comparison_col = [comparison_column] 
        else:
             self.comparison_col = comparison_column 
             
        self.treatment_map = treatment_map if treatment_map is not None else {}
        self.cat_cleaning_map = cat_cleaning_map if cat_cleaning_map is not None else {} 
        self.elbow_style = elbow_style 
        self.n_init = n_init 
        self.X_scaled = None
        self.results = {}
        self.best_method = None
        self.selected_method_from_viz = None
        self.X_final_features = None 
        self.out = Output()   
        self.max_k = 2

    def _normalize_metric(self, scores: pd.Series, ascending: bool = True) -> pd.Series:
        if scores.empty:
            return pd.Series(0, index=scores.index)
        ranks = rankdata(scores.fillna(scores.min() if ascending else scores.max()), method='dense')
        min_rank = ranks.min()
        max_rank = ranks.max()
        if max_rank == min_rank:
            return pd.Series(1.0, index=scores.index)
        normalized_ranks = (ranks - min_rank) / (max_rank - min_rank)
        if not ascending:
            return 1 - normalized_ranks
        else:
            return pd.Series(normalized_ranks, index=scores.index)

    def prepare_data(self):
        with self.out:
            clear_output()
            print("==================================================")
            print("➡️ SCHRITT 1: DATENVORBEREITUNG")
            X = self.df.drop(columns=self.comparison_col, errors='ignore') 
            X_cleaned = X.copy()
            num_cols = X_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_cleaned.select_dtypes(include=["object", "category", "bool"]).columns.difference(num_cols).tolist()
            
            for col in cat_cols.copy():
                if X_cleaned[col].nunique(dropna=True) == 2:
                    X_cleaned[col] = pd.factorize(X_cleaned[col])[0]
                    cat_cols.remove(col)
                    num_cols.append(col)
            
            for col in num_cols:
                if X_cleaned[col].isnull().any():
                    X_cleaned[col].fillna(X_cleaned[col].median(), inplace=True)
            
            for col in cat_cols:
                if X_cleaned[col].isnull().any():
                    X_cleaned[col].fillna(X_cleaned[col].mode()[0], inplace=True)

            X_final = pd.get_dummies(X_cleaned, columns=cat_cols, drop_first=True)
            self.X_final_features = X_final
            self.df = self.df.loc[X_final.index]
            
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X_final)
            print(f"✅ Daten fertig skaliert. Analysiert {self.X_scaled.shape[0]} Zeilen und {self.X_scaled.shape[1]} Features.")
            return True

    def show_elbow(self):
        with self.out:
            print("\n==================================================")
            print("➡️ SCHRITT 2: OPTIMALE K-BESTIMMUNG")
            distortions = []
            silhouette_scores = []
            self.max_k = min(11, len(self.X_scaled))
            K = range(2, self.max_k)
            
            for k in K:
                km = KMeans(n_clusters=k, random_state=42, n_init=self.n_init)
                km.fit(self.X_scaled)
                distortions.append(km.inertia_)
                silhouette_scores.append(silhouette_score(self.X_scaled, km.labels_))

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(K, distortions, self.elbow_style)
            plt.xlabel("Anzahl der Muster (k)")
            plt.ylabel("Distortion (Inertia)")
            plt.title("Ellbogen-Methode")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(K, silhouette_scores, "ro-")
            plt.xlabel("Anzahl der Muster (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette-Analyse")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            print(f"✅ Plots generiert. Bitte wählen Sie k.")

    def compare_methods(self, k: int):
        with self.out:
            print("\n==================================================")
            print(f"➡️ SCHRITT 3: METHODENVERGLEICH (k = {k})")
            self.results = {}
            methods_to_run = {
                "KMeans": KMeans(n_clusters=k, random_state=42, n_init=self.n_init),
                "GMM": GaussianMixture(n_components=k, random_state=42),
                "Agglomerative": AgglomerativeClustering(n_clusters=k),
                "Spectral": SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors'),
                "Birch": Birch(n_clusters=k)
            }

            for name, model in methods_to_run.items():
                try:
                    labels = model.fit_predict(self.X_scaled)
                    if len(set(labels)) > 1:
                        self.results[name] = {
                            "Labels": labels,
                            "Model": model,
                            "Silhouette": silhouette_score(self.X_scaled, labels),
                            "CH": calinski_harabasz_score(self.X_scaled, labels),
                            "DB": davies_bouldin_score(self.X_scaled, labels)
                        }
                        print(f"  - **{name}** erfolgreich berechnet.")
                except Exception as e:
                    print(f"  ❌ **{name}** Fehler: {e}")

            if not self.results:
                print("❌ Keine gültigen Ergebnisse.")
                return

            res_df = pd.DataFrame([{"Methode": name, **{m:v for m,v in metrics.items() if m not in ['Labels', 'Model']}} for name, metrics in self.results.items()]).set_index("Methode")
            res_df['Norm_Silhouette'] = self._normalize_metric(res_df['Silhouette'])
            res_df['Norm_CH'] = self._normalize_metric(res_df['CH'])
            res_df['Norm_DB'] = self._normalize_metric(res_df['DB'], ascending=False)
            res_df['Total Score'] = res_df[['Norm_Silhouette', 'Norm_CH', 'Norm_DB']].mean(axis=1)
            self.best_method = res_df['Total Score'].idxmax()
            self.selected_method_from_viz = self.best_method
            
            print("\n🏆 Metriken-Vergleich:")
            display(res_df.sort_values('Total Score', ascending=False))
            print(f"\n**Beste Methode:** **{self.best_method}**")

    def plot_clusters(self, method: str):
        with self.out:
            if method not in self.results:
                print(f"❌ Methode '{method}' nicht gefunden.")
                return

            labels = self.results[method]["Labels"]
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(self.X_scaled)
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=[str(l) for l in labels], palette="viridis", legend="full")
            plt.title(f"Muster: {method} (PCA-reduziert)")
            plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.show()

    def _calculate_and_plot_feature_importance(self, method: str):
        with self.out:
            if method not in self.results or "Model" not in self.results.get(method, {}):
                print(f"\n⚠️ Merkmalswichtigkeit für **{method}** übersprungen (Modell nicht gespeichert).")
                return

            try:
                model = self.results[method]["Model"]
                feature_names = self.X_final_features.columns.tolist()
                
                if hasattr(model, 'cluster_centers_'):
                    centers = model.cluster_centers_
                elif hasattr(model, 'means_'):
                    centers = model.means_
                else:
                    print(f"\n⚠️ Merkmalswichtigkeit für **{method}** übersprungen (Modell-Typ hat keine Zentren).")
                    return

                feature_std = centers.std(axis=0)
                df_importance = pd.DataFrame({'Merkmal': feature_names, 'Wichtigkeit': feature_std}).sort_values(by='Wichtigkeit', ascending=False).head(15)

                print("\n==================================================")
                print(f"💡 EINFLUSSREICHSTE MERKMALE ({method})")
                print("==================================================")
                plt.figure(figsize=(10, 7))
                sns.barplot(x='Wichtigkeit', y='Merkmal', data=df_importance, palette="viridis")
                plt.title(f"Top 15 Einflussreichste Merkmale ({method})")
                plt.xlabel("Wichtigkeit (Standardabweichung über Cluster-Zentren)")
                plt.ylabel("Merkmal")
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"\n❌ FEHLER bei der Berechnung der Merkmalswichtigkeit: {e}")

    def execute_assignment_and_comparison(self, solution_col_name: str):
        with self.out:
            assignment_method = self.selected_method_from_viz
            if assignment_method is None:
                print("❌ Keine Methode ausgewählt.")
                return

            print(f"\n==================================================")
            print(f"📊 SCHRITT 7: ZUORDNUNG & VERGLEICH ({assignment_method})")
            print("==================================================")
            
            labels = self.results[assignment_method]["Labels"]
            self.df['Cluster_ML'] = pd.Series(labels, index=self.df.index)
            print(f"✅ Spalte **'Cluster_ML'** in DataFrame eingefügt.")

            if solution_col_name and solution_col_name != "Kein Vergleich":
                if solution_col_name not in self.df.columns:
                    print(f"❌ Lösungsspalte '{solution_col_name}' nicht gefunden.")
                    return

                df_compare = self.df.dropna(subset=['Cluster_ML', solution_col_name])
                contingency = pd.crosstab(df_compare[solution_col_name], df_compare['Cluster_ML'])
                
                print("\n--- Kontingenztabelle ---")
                display(contingency)
                plt.figure(figsize=(8, 5))
                sns.heatmap(contingency, annot=True, fmt="d", cmap="viridis")
                plt.title(f"Verteilung: '{solution_col_name}' vs. Cluster")
                plt.show()

                self._calculate_and_plot_feature_importance(assignment_method)

            print("\n✅ Schritt 7 abgeschlossen.")

# (Rest der Klasse ClusteringTool bleibt hier unverändert, da sie die App-Logik steuert)

class ClusteringTool:
    def __init__(self):
        # ... (der bestehende Code der Tool-Klasse)
        pass # Platzhalter, da der Inhalt sehr lang ist und hier nicht wiederholt werden muss

# Führen Sie das interaktive Clustering-Tool aus
# ClusteringTool() # Auskommentiert, um nicht automatisch zu starten
