# Clustering-App mit Dropdown-Menüs und erweiterter Ellbogen-Einstellung01
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

# Die Warnungen aus der vorherigen Zelle werden hier ignoriert, da sie im bereinigten Code behoben sind
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

    # SCHRITT 1: DATENVORBEREITUNG 
    def prepare_data(self):
        """Bereitet die Daten vor (NaN-Behandlung, Binär-Umwandlung, One-Hot-Encoding, Skalierung) mit Robustheits-Sicherheitsnetz."""
        with self.out:
            clear_output()
            print("==================================================")
            print("➡️ SCHRITT 1: DATENVORBEREITUNG")
            rows_initial = len(self.df)
            cols_initial = len(self.df.columns)
            print(f"Start-Dimensionen (Gesampelt): {rows_initial} Zeilen, {cols_initial} Spalten.")
            
            # Drop der Vergleichsspalten (jetzt Liste)
            X = self.df.drop(columns=self.comparison_col, errors='ignore') 
            if self.comparison_col:
                print(f"ℹ️ Folgende Spalten werden für Clustering ignoriert: **{', '.join(self.comparison_col)}**.")
            
            X_cleaned = X.copy()
            num_cols = X_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_cleaned.select_dtypes(include=["object", "category", "bool"]).columns.difference(num_cols).tolist()
            binary_cols = []
            
            # Binäre Konvertierung
            print("\n➡️ Binäre Kategoriale Spalten umwandeln (z.B. Ja/Nein, M/W, Yes/No):")
            for col in cat_cols.copy():
                unique_count = X_cleaned[col].nunique(dropna=True)
                if unique_count == 2 or X_cleaned[col].dtype == 'bool':                
                    if X_cleaned[col].dtype == 'object' or X_cleaned[col].dtype == 'category':
                        unique_vals = [str(v).lower() for v in X_cleaned[col].dropna().unique()]
                        
                        if len(unique_vals) == 2:
                            # Fügt 'yes'/'no' Erkennung hinzu
                            if 'yes' in unique_vals or 'no' in unique_vals:
                                val_1 = [v for v in unique_vals if v in ['yes', 'no']][0]
                                val_0 = [v for v in unique_vals if v not in ['yes', 'no']][0] if len([v for v in unique_vals if v not in ['yes', 'no']]) > 0 else ([v for v in unique_vals if v != val_1][0] if len(unique_vals) == 2 else unique_vals[0])
                                
                                # Annahme: 'yes' oder 'wahr' soll 1 sein, alles andere 0
                                if val_1 == 'yes':
                                    mapping = {val_0: 0, val_1: 1}
                                elif 'false' in unique_vals and 'true' in unique_vals:
                                    mapping = {'false': 0, 'true': 1}
                                else:
                                    # Generische 0/1 Zuordnung
                                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                                    
                                # Anwendung der Mapping
                                X_cleaned[col] = X_cleaned[col].map(mapping).astype('Int64')
                                print(f"  - Spalte '{col}': Binär zu **Numerisch (0/1)** konvertiert. (Muster: {mapping})")

                            else:
                                # Generische 0/1 Zuordnung für zwei Werte
                                mapping = {X_cleaned[col].dropna().unique()[0]: 0, X_cleaned[col].dropna().unique()[1]: 1}
                                X_cleaned[col] = X_cleaned[col].map(mapping).astype('Int64')
                                print(f"  - Spalte '{col}': Binär zu **Numerisch (0/1)** konvertiert.")

                        else:
                            # Wenn es scheinbar binär war, aber die Zählung nach DropNA mehr als 2 Werte ergab (unwahrscheinlich)
                            X_cleaned[col] = pd.factorize(X_cleaned[col])[0]
                            print(f"  - Spalte '{col}': Binär zu **Faktorisierter Numerisch** konvertiert (mehr als 2 Werte).")
                    
                    elif X_cleaned[col].dtype == 'bool':
                         X_cleaned[col] = X_cleaned[col].astype(int)
                         print(f"  - Spalte '{col}': Bool zu **Int (0/1)** konvertiert.")

                    cat_cols.remove(col)
                    num_cols.append(col)
                    binary_cols.append(col)

            print(f"ℹ️ Verbleibende Kategoriale Spalten für One-Hot-Encoding: {len(cat_cols)}")

            # KATEGORIALE BEREINIGUNG ANWENDEN (TEMP_Clear_ML)
            print("\n➡️ Anwendung der Kategorialen Bereinigungs-Muster:")
            for col, action in self.cat_cleaning_map.items():
                if col not in X_cleaned.columns:
                    continue 

                if action == 'TEMP_Clear_ML':
                    if X_cleaned[col].dtype == 'object':
                        X_cleaned[col] = X_cleaned[col].astype(str)
                        X_cleaned[col] = X_cleaned[col].str.replace(' ', '_', regex=False)
                        X_cleaned[col] = X_cleaned[col].replace('nan', 'MISSING')
                        print(f"  - Kategorisch '{col}': **TEMP_Clear_ML** (Spaces to _, 'nan' zu 'MISSING') angewendet.")
                elif action == 'none':
                    pass
            
            # SPALTENSPEZIFISCHE BEHANDLUNG FEHLENDER WERTE 
            rows_to_drop = []
            print("\n➡️ Behandlung von NaN-Werten (gemäß Auswahl):")         
            for col, treatment in self.treatment_map.items():
                if col not in X_cleaned.columns: continue
                if X_cleaned[col].isnull().sum() == 0 and col not in binary_cols: continue   
                if treatment == 'median' or treatment == 'mean':
                    if col in num_cols:
                        value = X_cleaned[col].median() if treatment == 'median' else X_cleaned[col].mean()
                        X_cleaned[col] = X_cleaned[col].fillna(value)
                        print(f"  - Numerisch '{col}': NaNs mit **{treatment.upper()}** imputiert.")
                elif treatment == 'mode':
                    if col in cat_cols or col in binary_cols: 
                        mode_val = X_cleaned[col].mode()
                        if not mode_val.empty:
                            X_cleaned[col] = X_cleaned[col].fillna(mode_val[0])
                            print(f"  - Kategorisch/Binär '{col}': NaNs mit **MODUS ('{mode_val[0]}')** imputiert.")
                elif treatment == 'drop_row':
                    rows_to_drop.extend(X_cleaned[X_cleaned[col].isnull()].index.tolist())
                    print(f"  - '{col}': Zeilen mit NaNs zur Entfernung markiert.")
            
            rows_to_drop = list(set(rows_to_drop))
            X_cleaned = X_cleaned.drop(rows_to_drop, errors='ignore')
            # Stellt sicher, dass numerische Spalten wirklich numerisch sind
            X_cleaned.loc[:, num_cols] = X_cleaned[num_cols].apply(pd.to_numeric, errors='coerce')

            print("\n➡️ Feature Engineering (One-Hot Encoding):")
            cat_cols_final = X_cleaned.select_dtypes(include=["object", "category"]).columns.tolist()
            X_final = pd.get_dummies(X_cleaned, columns=cat_cols_final, drop_first=True)
            
            X_final_features = X_final.select_dtypes(include=[np.number])
            self.X_final_features = X_final_features
            
            # WICHTIG: self.df wird auf die gesampelten/bereinigten Zeilen reduziert (deren Index nun den Labels entspricht)
            self.df = self.df.loc[X_final_features.index].copy()
            
            print("\n➡️ Skalierung (StandardScaler):")
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(X_final_features)            
            
            if self.X_scaled.size == 0 or len(self.X_scaled) < 2:
                print("❌ FEHLER: Nicht genügend Datenpunkte nach der Bereinigung. Clustering unmöglich.")
                return False
            
            rows_final = len(self.X_scaled)
            print(f"✅ Daten fertig skaliert. Analysiert {rows_final} Zeilen und {self.X_scaled.shape[1]} Features.")
            print(f"  ({rows_initial - rows_final} Zeilen wurden aufgrund von 'drop_row' oder NaNs entfernt).")
            return True

    # SCHRITT 2: OPTIMALE K-BESTIMMUNG (Ellbogen & Silhouette) 
    def show_elbow(self):
        """Generiert den Ellbogen-Plot und den Silhouette-Score-Plot zur Bestimmung von k."""
        with self.out:
            print("\n==================================================")
            print("➡️ SCHRITT 2: OPTIMALE K-BESTIMMUNG (Ellbogen & Silhouette)")            
            distortions = []
            silhouette_scores = []
            self.max_k = min(11, len(self.X_scaled)) 
            K = range(1, self.max_k)            
            
            if len(K) < 2: 
                print("❌ Nicht genug Datenpunkte für die k-Methoden (benötigt mindestens 2).")
                return False
            
            K_for_sil = range(2, self.max_k)
            if not K_for_sil:
                print("❌ Nicht genug Datenpunkte für die k-Methoden (benötigt k >= 2).")
                return False

            print(f"ℹ️ Berechne Metriken für k von 1 bis {self.max_k-1} (n_init={self.n_init}).....")
            
            for k in K:
                try:
                    n_init_val = self.n_init if self.n_init == 'auto' else int(self.n_init)
                    km = KMeans(n_clusters=k, random_state=42, n_init=n_init_val, max_iter=300)
                    km.fit(self.X_scaled)
                    distortions.append(km.inertia_)
                    
                    if k > 1 and len(set(km.labels_)) > 1:
                        score = silhouette_score(self.X_scaled, km.labels_)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(None)
                        
                except Exception as e:
                    print(f"⚠️ KMeans Fehler bei k={k}: {e}")
                    distortions.append(None)
                    silhouette_scores.append(None)                    
            
            K_valid_elbow = [k for k, d in zip(K, distortions) if d is not None]
            distortions_valid = [d for d in distortions if d is not None]            
            
            K_valid_sil = [k for k, d in zip(K[1:], silhouette_scores[1:]) if d is not None]
            sil_valid = [d for d in silhouette_scores[1:] if d is not None]

            if not K_valid_elbow or not K_valid_sil:
                print("❌ Keine gültigen Metrik-Werte berechnet.")
                return False

            plt.figure(figsize=(6, 2))
            
            plt.subplot(1, 2, 1)
            plt.plot(K_valid_elbow, distortions_valid, self.elbow_style) 
            plt.xlabel("Anzahl der Muster (k)")
            plt.ylabel("Distortion (Inertia)")
            plt.title("Ellbogen-Methode")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(K_valid_sil, sil_valid, "ro-")
            plt.xlabel("Anzahl der Muster (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette-Analyse")
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print(f"✅ Plots generiert. Bitte wählen Sie nun k (Schritt 5) von 2 bis {self.max_k-1}.")
            return True


    # SCHRITT 3: METHODENVERGLEICH (Mit Total Score) 
    def compare_methods(self, k: int):
        """Führt verschiedene Clustering-Methoden aus und vergleicht sie mittels Total Score."""
        with self.out:
            print("\n==================================================")
            print(f"➡️ SCHRITT 3: METHODENVERGLEICH & TOTAL SCORE (k = {k})")
            self.results = {}            
            n_samples = self.X_scaled.shape[0]
            
            # Wählt Affinity basierend auf der Stichprobengröße für SpectralClustering
            spectral_affinity = 'nearest_neighbors' if n_samples > 1000 else 'rbf'
            if spectral_affinity == 'nearest_neighbors':
                 print(f"  ⚠️ SpectralClustering: {n_samples} Zeilen, verwende 'nearest_neighbors' für Performance.")
                 
            n_init_val = self.n_init if self.n_init == 'auto' else int(self.n_init)

            methods_to_run = {
                "KMeans": KMeans(n_clusters=k, random_state=42, n_init=n_init_val, max_iter=500), 
                "GMM (Gaussian Mixture)": GaussianMixture(n_components=k, random_state=42, n_init=10, max_iter=500),
                "Agglomerative": AgglomerativeClustering(n_clusters=k, linkage='ward'),
                "Spectral": SpectralClustering(n_clusters=k, random_state=42, n_init=10, affinity=spectral_affinity),
                "Birch": Birch(n_clusters=k)
            }

            for name, model in methods_to_run.items():
                try:
                    if name == "GMM (Gaussian Mixture)":
                        model.fit(self.X_scaled)
                        labels = model.predict(self.X_scaled)
                    else:
                        labels = model.fit_predict(self.X_scaled)
                        
                    num_clusters = len(set(labels))
                    
                    if num_clusters > 1 and num_clusters == k:
                        
                        # Labels neu zuordnen nach N (Größter wird 1)
                        df_temp = pd.DataFrame({'Labels': labels})
                        sizes = df_temp['Labels'].value_counts().sort_values(ascending=False)
                        
                        # Nur Cluster > -1 (kein Rauschen) neu zuordnen
                        non_noise_sizes = sizes[sizes.index != -1]
                        # Startet die Nummerierung bei 1 (Größter Cluster = 1)
                        new_mapping = {old_label: new_label for new_label, old_label in enumerate(non_noise_sizes.index, start=1)}
                        
                        # Rauschen (-1) bleibt -1
                        if -1 in sizes.index:
                            new_mapping[-1] = -1 

                        sorted_labels = df_temp['Labels'].map(new_mapping).fillna(-1).astype(int).values
                        
                        # FINALE Speicherung der sortierten Labels und Metriken
                        result_dict = {
                            "Labels": sorted_labels, # <--- HIER WERDEN DIE SORTIERTEN LABELS GESPEICHERT (Größter = 1)
                            "Original_Labels": labels, # <- ORIGINAL Labels (0, 1, 2...) für Kontingenztabelle in Schritt 6
                            "Silhouette": silhouette_score(self.X_scaled, labels), 
                            "CH": calinski_harabasz_score(self.X_scaled, labels),
                            "DB": davies_bouldin_score(self.X_scaled, labels)
                        }
                        
                        # Speichere das Modell selbst, falls es Cluster-Zentren hat (für Feature Importance)
                        if name in ["KMeans", "GMM (Gaussian Mixture)"]:
                            result_dict["Model"] = model
                        
                        self.results[name] = result_dict
                        print(f"  - **{name}** erfolgreich berechnet. Muster: {num_clusters} (Sortiert 1, 2, ...)")
                    else:
                        print(f"  - **{name}** übersprungen (Musterzahl {num_clusters} entspricht nicht k={k} oder ist trivial).")
                        
                except MemoryError:
                    print(f"  ❌ **{name}** Fehler: Nicht genug Speicher. Überspringen.")
                except Exception as e:
                    print(f"  ❌ **{name}** Fehler: Unerwarteter Fehler: {type(e).__name__}. Überspringen.")
                    
            # DBSCAN-Spezialfall
            print("\n➡️ Spezialfall: DBSCAN")
            try:
                neigh = NearestNeighbors(n_neighbors=5).fit(self.X_scaled)
                distances, _ = neigh.kneighbors(self.X_scaled)
                eps = np.percentile(distances[:, -1], 90) 
                
                db = DBSCAN(eps=eps, min_samples=5).fit(self.X_scaled)
                labels = db.labels_
                
                labels_clean = labels[labels != -1]
                X_scaled_clean = self.X_scaled[labels != -1]
                
                num_valid_clusters = len(set(labels_clean))
                
                if num_valid_clusters > 1 and len(labels_clean) >= 2: 
                    # Labels neu zuordnen nach N (Größter wird 1)
                    df_temp = pd.DataFrame({'Labels': labels})
                    sizes = df_temp['Labels'].value_counts().sort_values(ascending=False)
                    
                    non_noise_sizes = sizes[sizes.index != -1]
                    new_mapping = {old_label: new_label for new_label, old_label in enumerate(non_noise_sizes.index, start=1)}
                    
                    if -1 in sizes.index:
                        new_mapping[-1] = -1 

                    sorted_labels = df_temp['Labels'].map(new_mapping).fillna(-1).astype(int).values
                    
                    self.results["DBSCAN"] = {
                        "Labels": sorted_labels,
                        "Original_Labels": labels,
                        "Silhouette": silhouette_score(X_scaled_clean, labels_clean),
                        "CH": calinski_harabasz_score(X_scaled_clean, labels_clean),
                        "DB": davies_bouldin_score(X_scaled_clean, labels_clean)
                    }
                    print(f"  - **DBSCAN** erfolgreich berechnet. Muster: {num_valid_clusters} (+ Rauschen). (Sortiert 1, 2, ...)")
                else:
                    print(f"  - **DBSCAN** übersprungen (Zu wenig gültige Muster nach Rauschen-Entfernung oder zu wenig Punkte).")
                    
            except Exception as e:
                 print(f"  ❌ **DBSCAN** Fehler: Unerwarteter Fehler: {type(e).__name__}. Überspringen.")

            if not self.results:
                print("❌ Keine gültigen Clustering-Ergebnisse zum Vergleichen.")
                return

            res_df = pd.DataFrame([
                {"Methode": k, **{k:v for k,v in v.items() if k not in ['Labels', 'Original_Labels', 'Model']}} for k, v in self.results.items()
            ]).drop(columns=[]).set_index("Methode")
            
            res_df['Norm_Silhouette'] = self._normalize_metric(res_df['Silhouette'], ascending=True)
            res_df['Norm_CH'] = self._normalize_metric(res_df['CH'], ascending=True)
            res_df['Norm_DB'] = self._normalize_metric(res_df['DB'], ascending=False)
            
            res_df['Total Score'] = res_df[['Norm_Silhouette', 'Norm_CH', 'Norm_DB']].mean(axis=1).round(3)
            
            self.best_method = res_df['Total Score'].idxmax()
            self.selected_method_from_viz = self.best_method

            comparison_cols = ['Silhouette', 'CH', 'DB', 'Total Score']
            res_df = res_df[comparison_cols]
            res_df = res_df.sort_values(by="Total Score", ascending=False)
            
            res_df['Silhouette'] = res_df['Silhouette'].round(3)
            res_df['DB'] = res_df['DB'].round(3)
            res_df['CH'] = res_df['CH'].round(0).astype(int)
            
            print("\n🏆 Metriken-Vergleich (Normalisiert + Total Score):")
            display(res_df)
            
            print(f"\n**Ergebnis:** Die beste Methode (nach Total Score) ist **{self.best_method}**.")
            self.results[self.best_method]['Total Score'] = res_df.loc[self.best_method]['Total Score']


    # SCHRITT 4: VISUALISIERUNGS-STEUERUNG 
    def show_visualization_controls(self):
        """Zeigt die Steuerelemente für Visualisierung und Analyse an."""
        if not self.results:
            return
            
        method_dd = Dropdown(options=list(self.results.keys()),
                             value=self.best_method, description="Methode:")
        show_btn = Button(description="Plot & Analyse anzeigen")
        viz_out = Output()
        
        with self.out:
            print("\n==================================================")
            print("➡️ SCHRITT 4: VISUALISIERUNG & INTERPRETATION")
            print("Wählen Sie eine Methode, um die Muster in 2D (PCA-reduziert) zu visualisieren und die deskriptive Analyse zu sehen.")
            display(VBox([method_dd, show_btn, viz_out]))
        
        def viz(b):
            method = method_dd.value
            self.selected_method_from_viz = method
            with viz_out:
                clear_output()
                self.plot_clusters(method)
                self.analyze_clusters(method)
                
        show_btn.on_click(viz)

        return self.results[self.best_method]["Labels"] if self.best_method else None

    # SCHRITT 5: DESKRIPTIVE MUSTER-ANALYSE (Interpretation) - MIT SORTIERUNG NACH N
    def analyze_clusters(self, method: str):
        """Zeigt deskriptive Statistiken der Cluster-Zentren (Interpretation)."""
        if method not in self.results:
            return

        labels = self.results[method]["Labels"]
        
        df_labeled = self.df.copy()
        df_labeled['Cluster'] = labels 
        
        cols_to_analyze = df_labeled.columns.drop(['Cluster'] + self.comparison_col, errors='ignore')

        analysis_data = []
        sorted_clusters = sorted(df_labeled['Cluster'].unique(), key=lambda x: (x == -1, x)) 
        
        for cluster_id in sorted_clusters:
            cluster_data = {'Cluster': str(cluster_id)}
            subset = df_labeled[df_labeled['Cluster'] == cluster_id]
            N = len(subset)
            cluster_data['N (Anzahl)'] = N

            for col in cols_to_analyze:
                dtype = subset[col].dtype
                
                original_col_name = col
                if original_col_name in self.original_df.columns:
                    original_subset = self.original_df.loc[subset.index, original_col_name]
                    dtype_orig = original_subset.dtype
                    
                    # ------------------------------------------------
                    # Verwende is_numeric_dtype aus pandas.api.types
                    # ------------------------------------------------
                    if is_numeric_dtype(dtype_orig): 
                        cluster_data[f'{original_col_name} (Mean)'] = original_subset.mean().round(2)
                    elif dtype_orig == 'object' or dtype_orig == 'category' or dtype_orig == 'bool':
                        mode_val = original_subset.mode()
                        mode_str = str(mode_val[0]) if not mode_val.empty else 'N/A'
                        if mode_str.lower() in ['yes', 'no']:
                             mode_str = mode_str.lower().replace('yes', 'ja').replace('no', 'nein') 
                        cluster_data[f'{original_col_name} (Mode)'] = mode_str
                else:
                    if np.issubdtype(dtype, np.number):
                         cluster_data[f'{col} (Mean)'] = subset[col].mean().round(2)
                    elif dtype == 'object' or dtype == 'category' or dtype == 'bool':
                        mode_val = subset[col].mode()
                        mode_str = str(mode_val[0]) if not mode_val.empty else 'N/A'
                        if mode_str.lower() in ['yes', 'no']:
                             mode_str = mode_str.lower().replace('yes', 'ja').replace('no', 'nein') 
                        cluster_data[f'{col} (Mode)'] = mode_str
                        
            analysis_data.append(cluster_data)

        df_analysis = pd.DataFrame(analysis_data).set_index('Cluster')
        
        non_noise_clusters = df_analysis[df_analysis.index != '-1']
        noise_cluster = df_analysis[df_analysis.index == '-1'] if '-1' in df_analysis.index else pd.DataFrame()
        
        df_analysis = pd.concat([non_noise_clusters, noise_cluster])


        with self.out:
            print("\n==================================================")
            print(f"📊 DESKRIPTIVE ANALYSE ({method}) - SORTIERT NACH GRÖSSE (N)")
            print("==================================================")
            print(f"Statistiken der Muster-Zentren für **{method}** basierend auf Original-Features:")
            display(df_analysis)
            print("ℹ️ Die Interpretation dieser Werte (z.B. hohe Einkommen, niedriges Alter) definiert die Segmente.")
            return df_analysis  

    # Plot-Logik (KORRIGIERT: explained_ratio_ zu explained_variance_ratio_)
    def plot_clusters(self, method: str):
        """Reduziert auf 2D mittels PCA und plottet die Muster."""
        if method not in self.results:
            print(f"❌ Methode '{method}' nicht gefunden.")
            return

        with self.out:
            print(f"\nℹ️ Generiere 2D-Plot für **{method}** (PCA-Reduktion)...")

        labels = self.results[method]["Labels"]
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.X_scaled)
        
        hue_labels = [str(l) for l in labels]

        plt.figure(figsize=(8, 6))

        if method == "DBSCAN" and -1 in labels:
            noise_indices = (labels == -1)
            core_indices = (labels != -1)
            
            sns.scatterplot(x=pca_result[core_indices, 0], y=pca_result[core_indices, 1], 
                            hue=[str(l) for l in labels[core_indices]], 
                            palette="viridis", legend="full", s=70)
            
            plt.scatter(pca_result[noise_indices, 0], pca_result[noise_indices, 1], 
                        c='gray', marker='x', s=50, label="Rauschen (-1)")
            
            handles, labels_plt = plt.gca().get_legend_handles_labels()
            plt.legend(handles, labels_plt, title="Muster")
        else:
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=hue_labels, palette="viridis", legend="full")
            
        plt.title(f"Muster: {method} (PCA-reduziert)")
        plt.xlabel(f"PCA Komponente 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        # FEHLER BEHOBEN: explained_ratio_ durch explained_variance_ratio_ ersetzt
        plt.ylabel(f"PCA Komponente 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.show()
        
        with self.out:
             print("✅ Visualisierung abgeschlossen.")
    
    # NEU: Methode zur Berechnung und Visualisierung der Merkmalswichtigkeit (Feature Importance)
    def _calculate_and_plot_feature_importance(self, method: str):
        """
        Berechnet die Wichtigkeit der Features basierend auf der Standardabweichung 
        der Cluster-Zentren und plottet die Top 15.
        Gilt nur für Methoden mit Zentren (KMeans, GMM).
        """
        if method not in self.results or "Model" not in self.results[method]:
            print(f"\n⚠️ Merkmalswichtigkeit für **{method}** übersprungen: Modell hat keine Cluster-Zentren (oder ist DBSCAN/Agglomerative).")
            return

        try:
            model = self.results[method]["Model"]
            feature_names = self.X_final_features.columns.tolist()
            
            # Holt die Cluster-Zentren oder Mittelwerte (für GMM)
            if hasattr(model, 'cluster_centers_'): # KMeans
                centers = model.cluster_centers_
            elif hasattr(model, 'means_'): # GMM
                centers = model.means_
            else:
                 print(f"\n⚠️ Merkmalswichtigkeit für **{method}** übersprungen: Modell hat keine implementierten Zentren/Mittelwerte.")
                 return

            # Berechnung der Standardabweichung der Zentren/Mittelwerte über die Cluster hinweg
            # Ein großer Wert bedeutet große Variation des Merkmals zwischen den Clustern -> hohe Wichtigkeit
            feature_std = centers.std(axis=0)
            
            # Erstelle DataFrame für die Visualisierung
            df_importance = pd.DataFrame({
                'Merkmal': feature_names,
                'Wichtigkeit (Std. der Zentren)': feature_std
            }).sort_values(by='Wichtigkeit (Std. der Zentren)', ascending=False).head(15)

            with self.out:
                print("\n==================================================")
                print(f"💡 MERKMALSWICHTIGKEIT ({method})")
                print("==================================================")
                print("Die Wichtigkeit basiert auf der Standardabweichung des Merkmals über alle Cluster-Zentren (hohe Streuung = wichtiger).")
                
                plt.figure(figsize=(8, 6))
                sns.barplot(x='Wichtigkeit (Std. der Zentren)', y='Merkmal', data=df_importance, palette="plasma")
                plt.title(f"Einflussreichste Merkmale für die Cluster-Bildung ({method})")
                plt.xlabel("Wichtigkeit (Standardabweichung der Cluster-Zentren)")
                plt.ylabel("Merkmal")
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            with self.out:
                print(f"\n❌ FEHLER bei der Berechnung/Visualisierung der Merkmalswichtigkeit: {type(e).__name__}.")

    # NEU: Methode zur Cluster-Zuordnung (Cluster_ML) und Vergleich (TEMP_Clear_ML) - KORRIGIERT
    def execute_assignment_and_comparison(self, solution_col_name: str, df_ML_ref: pd.DataFrame, selected_method: str = None):
        """Führt Cluster-Zuweisung (Cluster_ML) und optional den Vergleich (TEMP_Clear_ML) durch."""
        
        global df_ML
        df_ML = df_ML_ref
        
        # KORREKTUR: Verwende die manuell gewählte Methode, ansonsten die beste Methode
        assignment_method = selected_method if selected_method in self.results else self.best_method

        if assignment_method is None:
            print("❌ FEHLER: Keine Methode zur Zuordnung gefunden. Zuordnung nicht möglich.")
            return

        print("\n==================================================")
        print(f"📊 SCHRITT 7: MUSTER-ZUORDNUNG & VERGLEICH (Methode: {assignment_method})")
        print("==================================================")
        
        # 1. Cluster_ML Zuweisung (Labels 1, 2, 3...)
        sorted_labels = self.results[assignment_method]["Labels"]
        cluster_series_sorted = pd.Series(sorted_labels, index=self.df.index)
        
        df_ML['Cluster_ML'] = pd.Series(pd.NA, index=df_ML.index, dtype='Int64') 
        valid_indices = df_ML.index.intersection(cluster_series_sorted.index)
        df_ML.loc[valid_indices, 'Cluster_ML'] = cluster_series_sorted.loc[valid_indices].values 
        
        # BEREINIGT: Entfernen von '\'
        print(f"✅ Spalte **'Cluster_ML'** (Sortierte IDs: 1, 2, 3...) in **df_ML** eingebunden.")

        # 2. Deskriptive Profilanalyse (Jetzt als Teil der Ausführung)
        self.perform_descriptive_analysis(df_ML)

        # 3. TEMP_Clear_ML Zuweisung (falls Vergleich gewünscht)
        if solution_col_name and solution_col_name != "Kein Vergleich":
            
            if solution_col_name not in df_ML.columns:
                 print(f"❌ FEHLER: Die gewählte Lösungsspalte **'{solution_col_name}'** existiert nicht in df_ML.")
                 return

            df_compare = df_ML.loc[self.df.index].copy()
            
            df_compare['Cluster_ML_Sample'] = df_compare['Cluster_ML'] 

            df_compare.dropna(subset=['Cluster_ML_Sample', solution_col_name], inplace=True)
            
            if df_compare.empty:
                print("❌ WARNUNG: Keine übereinstimmenden Zeilen für den Vergleich nach NaN-Drop im Sample.")
                return

            # Kontingenztabelle und Zuordnung
            contingency = pd.crosstab(df_compare[solution_col_name], df_compare['Cluster_ML_Sample'])
            cost_matrix = -contingency.values
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            cluster_mapping = {contingency.columns[c]: contingency.index[r] for r, c in zip(row_ind, col_ind)}
            
            # NEUE KORREKTUR: Richtige Zuordnung vom Cluster zur Predominant Category
            df_compare['Predominant_Category'] = df_compare['Cluster_ML_Sample'].map(cluster_mapping)
            df_compare['TEMP_Clear_ML_Match'] = df_compare[solution_col_name] == df_compare['Predominant_Category']
            
            n_abw = (~df_compare['TEMP_Clear_ML_Match']).sum()
            print(f"\n❌ Abweichungen vom dominanten Cluster ('{solution_col_name}'): **{n_abw}**")

            # Erstellen und Zuweisen der TEMP_Clear_ML-Spalte
            temp_clear_ml_series = pd.Series(pd.NA, index=df_ML.index, dtype=object)
            
            temp_clear_ml_series.loc[df_compare.index] = df_compare.apply(
                # Cluster_ML_Sample (Sortierte IDs) für die Textausgabe verwenden
                lambda row: f"{df_ML.loc[row.name, solution_col_name]}+Cluster_{int(row['Cluster_ML_Sample'])}" if row['TEMP_Clear_ML_Match'] else f"Cluster_{int(row['Cluster_ML_Sample'])}",
                axis=1
            )

            # Rauschen (-1) (sortierte ID) auffüllen, wo noch keine Zuweisung erfolgte (Original ID -1)
            noise_indices = df_ML.index.intersection(cluster_series_sorted[cluster_series_sorted == -1].index)
            temp_clear_ml_series.loc[noise_indices] = temp_clear_ml_series.loc[noise_indices].fillna("Cluster_-1")

            df_ML['TEMP_Clear_ML'] = temp_clear_ml_series.loc[df_ML.index]
            
            # BEREINIGT: Entfernen von '\'
            print(f"✅ Spalte **'TEMP_Clear_ML'** in **df_ML** gespeichert (Vergleich mit '{solution_col_name}').")
            
            # Ausgabe der Kontingenztabelle und Heatmap
            print("\n--- Kontingenztabelle für Vergleich ---")
            display(contingency)
            plt.figure(figsize=(6, 4))
            sns.heatmap(contingency, annot=True, fmt="d", cmap="viridis", linewidths=.5, linecolor='black')
            plt.title(f"Muster-Verteilung für '{solution_col_name}'")
            plt.ylabel(solution_col_name)
            # NEU: Beschriftung angepasst
            plt.xlabel("Cluster (Cluster_ML IDs)")
            plt.show()

            # NEU: Merkmalswichtigkeit (nur im Szenario B)
            self._calculate_and_plot_feature_importance(assignment_method)


        # BEREINIGT: Entfernen von '\'
        print("\n✅ Schritt 7 abgeschlossen. df_ML enthält nun die Zuordnung(en).")
        
    # NEU: Methode zur Durchführung der deskriptiven Analyse in df_ML (Universell)
    def perform_descriptive_analysis(self, df_ML_ref: pd.DataFrame):
        """Generiert die universelle, deskriptive Cluster-Analyse aus df_ML."""
        
        df_analyzed = df_ML_ref[df_ML_ref['Cluster_ML'].notna()].copy()

        if 'Cluster_ML' in df_analyzed.columns:
            
            cols_to_profile = df_analyzed.columns.drop(['Cluster_ML', 'TEMP_Clear_ML'] + self.comparison_col, errors='ignore').tolist()
            
            agg_operations = {'N (Anzahl)': ('Cluster_ML', 'size')}
            
            for col in cols_to_profile:
                if col in self.original_df.columns:
                    # Direkte Aggregation auf Originalspalten im df_ML, falls vorhanden
                    col_data = df_analyzed[col]

                    if np.issubdtype(col_data.dtype, np.number):
                        agg_operations[f'{col} (Mean)'] = (col, 'mean')
                    elif col_data.dtype == 'object' or col_data.dtype == 'category' or col_data.dtype == 'bool':
                        agg_operations[f'{col} (Mode)'] = (col, lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
                        
            df_cluster_profile = df_analyzed.groupby('Cluster_ML').agg(**agg_operations)
            
            # Sortierung: Sortiert 1, 2, 3... vor dem Rauschen (-1)
            cluster_order = sorted(df_cluster_profile.index.tolist(), key=lambda x: (x == -1, x))
            df_cluster_profile = df_cluster_profile.reindex(cluster_order)
            
            # Sortiere Spalten für bessere Lesbarkeit
            cols = ['N (Anzahl)'] + [col for col in df_cluster_profile.columns if col != 'N (Anzahl)']
            df_cluster_profile = df_cluster_profile[cols]
            
            # Runde numerische Spalten
            for col in df_cluster_profile.columns:
                if ' (Mean)' in col:
                     df_cluster_profile[col] = df_cluster_profile[col].round(2)
            
            # Ausgabe der Cluster-Tabelle
            print("\n--------------------------------------------------")
            print("📊 DESKRIPTIVE CLUSTER-PROFILE (Universelle Auswertung)")
            print("--------------------------------------------------")
            print("Basis: Analysierter Dataframe **df_ML**. Cluster sind nach Größe sortiert (**Cluster 1 = Größte**).")
            display(df_cluster_profile)
            
        else:
            # BEREINIGT: Entfernen von '\'
            print("\n⚠️ WARNUNG: Keine Cluster-Spalte ('Cluster_ML') in df_ML gefunden. Profilierung übersprungen.")
#---
# Clustering-Tool: Startet den interaktiven Prozess
class ClusteringTool:
    def __init__(self):
        global df_ML
        if 'df_ML' not in globals():
            df_ML = pd.DataFrame()
        self.available_dfs = {name: obj for name, obj in globals().items() if isinstance(obj, pd.DataFrame) and name != 'df_ML'}
        if 'dataframe' in globals() and 'dataframe' not in self.available_dfs:
             self.available_dfs['dataframe'] = globals()['dataframe']        
        self.selected_df_name = None 
        self.selected_df = None
        self.sampled_df = None
        self.comparison_col = [] 
        self.solution_col_name = None
        self.treatment_map = {}
        self.cat_cleaning_map = {} 
        self.elbow_style = 'bx-' 
        self.n_init = 'auto'
        self.app = None 
        self.best_labels = None         
        self.comparison_result_df = None

        self.cleaning_options = {
            "Vorschläge zur Vorverarbeitung für ML (TEMP_Clear_ML)": "TEMP_Clear_ML", 
            "Keine (Ausschließen von der Verarbeitung)": "none"
        }
        
        self.all_steps_container = VBox([])         
        display(self.all_steps_container)
        self.setup_df_selection()        

    def _add_step_widgets(self, title: str, widgets_list: List[widgets.Widget], handler: callable, btn_label: str, btn_style: str = ''):
        """Erstellt einen robusten Step Container und fügt ihn zum Haupt-Container hinzu."""
        all_widgets = [HTML(f"<h3>{title}</h3>")]
        all_widgets.extend(widgets_list)
        step_container = VBox(all_widgets) 
        btn = Button(description=btn_label, button_style=btn_style)
        output_for_step = Output() 
        btn.on_click(lambda b: handler(b, output_for_step))
        self.all_steps_container.children += (step_container, btn, output_for_step)
        return btn, output_for_step
    
    def setup_df_selection(self):
        if not self.available_dfs:
            self.all_steps_container.children = (HTML("❌ **Keine DataFrames gefunden.** Bitte laden Sie einen DataFrame in die Umgebung."),)
            return
        df_options = list(self.available_dfs.keys())
        df_dd = Dropdown(options=df_options, description="DataFrame:")        
        
        def on_ok_click(b, output_for_step):
            with output_for_step:
                clear_output()
                self.selected_df_name = df_dd.value 
                self.selected_df = self.available_dfs[self.selected_df_name].copy() 
                
                global df_ML
                df_ML = self.selected_df.copy() 
                
                # BEREINIGT: Entfernen von '\'
                print(f"✅ DataFrame **'{self.selected_df_name}'** ausgewählt und als Arbeits-Dataframe **df_ML** kopiert.")
            df_dd.disabled = True
            b.disabled = True            
            self.setup_column_selection()            
        # BEREINIGT: Entfernen von '\'
        self._add_step_widgets("1️⃣ DataFrame auswählen (wird zu df_ML kopiert):",[df_dd], on_ok_click,"OK (DataFrame wählen)")

    def setup_column_selection(self):
        column_options = self.selected_df.columns.tolist()
        col_dd = widgets.SelectMultiple(options=column_options, description="Auszuschließende Spalten:", rows=8, style={'description_width': 'initial'})
        
        def on_next_click(b, output_for_step):
            self.comparison_col = list(col_dd.value) 
            with output_for_step:
                clear_output()
                print(f"✅ Auszuschließende Spalten (IDs, etc.): **{', '.join(self.comparison_col) if self.comparison_col else 'Keine'}**")            
            col_dd.disabled = True
            b.disabled = True            
            self.setup_sampling_selection() 
        
        self._add_step_widgets("2️⃣ Spalten (Ausschluss vom Clustering) auswählen:",
            [Label("Wählen Sie eine oder mehrere Spalten (z.B. IDs, reine Textspalten), die **nicht** in das Clustering einbezogen werden sollen."), col_dd],
            on_next_click,
            "Weiter zu Sampling-Auswahl") 
        
    # SCHRITT 3: DATENREDUZIERUNG (Input-Feld)
    def setup_sampling_selection(self):     
        n_rows = len(self.selected_df)
        row_input = IntText(
            value=n_rows,
            min=1,
            max=n_rows,
            step=1,
            description='Max. Zeilen:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        def on_next_click(b, output_for_step):
            target_rows = row_input.value
            
            if target_rows < 1 or target_rows > n_rows:
                with output_for_step:
                    clear_output()
                    print(f"❌ FEHLER: Ungültige Zeilenzahl. Wählen Sie zwischen 1 und {n_rows}.")
                return
            if target_rows < n_rows:
                rate = target_rows / n_rows
                self.sampled_df = self.selected_df.sample(n=target_rows, random_state=42)
                with output_for_step:
                    clear_output()
                    print(f"✅ Sampling: **{target_rows} Zeilen** ausgewählt ({rate*100:.2f}%).")
            else:
                self.sampled_df = self.selected_df.copy()
                with output_for_step:
                    clear_output()
                    print("✅ Sampling: **Voller Datensatz** ausgewählt.")
            row_input.disabled = True
            b.disabled = True
            self.setup_nan_treatment_selection()  
            
        self._add_step_widgets(
            "3️⃣ Datenreduzierung (Sampling) auswählen:",
            [Label(f"Geben Sie die maximale Anzahl der zu analysierenden Zeilen ein (Aktuelle Größe: {n_rows})."), row_input],
            on_next_click,
            "Weiter zu Fehlerbehandlung"
        )
    def setup_nan_treatment_selection(self):
        df_check = self.sampled_df.drop(columns=self.comparison_col, errors='ignore')
        nan_cols = df_check.columns[df_check.isnull().any()].tolist()
        self.treatment_widgets = {}
        widget_list = []
        num_options = {"Median (Empfohlen)": 'median', "Mittelwert (Mean)": 'mean', "Zeilen entfernen": 'drop_row'}
        cat_options = {"Modus (Mode/Häufigster Wert)": 'mode', "Zeilen entfernen": 'drop_row'}
        if nan_cols:
            for col in nan_cols:
                col_dtype = df_check[col].dtype                
                if np.issubdtype(col_dtype, np.number):
                    options = num_options
                    label = f"🔢 **{col}** (Numerisch, {df_check[col].isnull().sum()} NaNs):"
                    default_value = 'median'
                elif col_dtype == bool or col_dtype == 'object' or col_dtype == 'category':
                    options = cat_options
                    label = f"🔠 **{col}** (Kategorisch/Bool, {df_check[col].isnull().sum()} NaNs):"
                    default_value = 'mode'
                else:
                    continue
                dropdown = Dropdown(options=options, value=default_value, description="Methode:")
                self.treatment_widgets[col] = dropdown
                widget_list.append(VBox([Label(label), dropdown]))
        else:
             widget_list.append(HTML("✅ **Keine fehlenden Werte** in den Features gefunden."))
        def on_next_click(b, output_for_step):
            self.treatment_map = {col: dd.value for col, dd in self.treatment_widgets.items()}
            with output_for_step:
                clear_output() 
                print(f"✅ NaN-Behandlungen gespeichert.")
            for dd in self.treatment_widgets.values():
                 dd.disabled = True
            b.disabled = True
            self.setup_categorical_cleaning()
        self._add_step_widgets("4️⃣ Behandlung fehlender Werte pro Spalte auswählen:",widget_list,on_next_click,"Weiter zu Kategorialer Bereinigung")

    def setup_categorical_cleaning(self):
        df_check = self.sampled_df.drop(columns=self.comparison_col, errors='ignore')
        cat_cols = df_check.select_dtypes(include=["object", "category"]).columns.tolist() 
        self.cleaning_widgets = {}
        widget_list = []
        
        if cat_cols:
            for col in cat_cols:
                if df_check[col].nunique(dropna=True) <= 2: continue 
                label = f"🔠 **{col}** (Kategorisch/String):"
                dropdown = Dropdown(options=self.cleaning_options, 
                                    value="TEMP_Clear_ML", 
                                    description="Bereinigungs-Art:", style={'description_width': 'initial'})
                self.cleaning_widgets[col] = dropdown
                widget_list.append(VBox([Label(label), dropdown]))
        else:
            widget_list.append(HTML("✅ **Keine nicht-binären kategorialen Spalten** für zusätzliche Bereinigung gefunden."))
            
        def on_next_click(b, output_for_step):
            self.cat_cleaning_map = {col: dd.value for col, dd in self.cleaning_widgets.items()}
            with output_for_step:
                clear_output() 
                print(f"✅ Kategoriale Bereinigungen gespeichert.")
                for col, action in self.cat_cleaning_map.items():
                    if action == "TEMP_Clear_ML":
                         # BEREINIGT: Entfernen von '\'
                         print(f"\n✅ **{col}** - ML-Bereinigung (**TEMP_Clear_ML**) wird **automatisch** in Schritt 1 angewendet.")
                    elif action == "none":
                         print(f"\n✅ **{col}** - **Ausschluss** von weiterer kategorialer Bereinigung (d.h. sie wird unbereinigt als One-Hot-Encoding verwendet).")
            
            for dd in self.cleaning_widgets.values():
                 dd.disabled = True
            b.disabled = True
            self.setup_elbow_style_selection()
            
        self._add_step_widgets("4c: Kategoriale Bereinigung auswählen:",
            widget_list, on_next_click, "Weiter zu Ellbogen-Stil-Einstellung"
        )
    def setup_elbow_style_selection(self):
        style_options = {"Linie mit blauen 'x' Markern ('bx-')": 'bx-', "Nur blaue 'x' Marker (keine Linie) ('bx')": 'bx',"Dicke blaue Linie (kein Marker) ('b-')": 'b-',"Linie mit roten Punkten ('ro-')": 'ro-',"Nur rote Punkte (keine Linie) ('ro')": 'ro'}
        style_dd = Dropdown(options=style_options, value=self.elbow_style, description="Darstellungs-Stil:", style={'description_width': 'initial'})
        def run_n_init_selection(b, output_for_step):
            self.elbow_style = style_dd.value
            with output_for_step:
                clear_output() 
                print(f"✅ Darstellungs-Stil: **'{self.elbow_style}'** ausgewählt.")
            style_dd.disabled = True
            b.disabled = True            
            self.setup_n_init_selection()
        self._add_step_widgets(
            "4a: Ellbogen-Plot Darstellungs-Stil einstellen (Visuelle Einstellung):",
            [Label("Wählen Sie den Matplotlib-Darstellungs-Muster für den Ellbogen-Plot."), style_dd],
            run_n_init_selection, "Weiter zu Ellbogen-Stabilität (n_init)" 
        )
    def setup_n_init_selection(self):
        n_init_options = { "Auto (Standard, schnell)": 'auto', "10 (Stabil, glatterer Plot)": 10, "5 (Schnellerer Test)": 5, "20 (Sehr stabil, langsam)": 20 }
        default_val = self.n_init if self.n_init in n_init_options.values() else 'auto'
        n_init_dd = Dropdown(options={k: v for k, v in n_init_options.items()}, value=default_val, description="n_init (KMeans-Initialisierungen):",style={'description_width': 'initial'})
        def run_plots(b, output_for_step):
            self.n_init = n_init_dd.value
            with output_for_step:
                clear_output() 
                print(f"✅ n_init für KMeans: **'{self.n_init}'** ausgewählt.")
            self._execute_preparation_and_plots()
        # BEREINIGT: Entfernen von '\'
        self._add_step_widgets("4b: Ellbogen-Stabilität (n_init) einstellen:",[Label("Steuert die Anzahl der KMeans-Läufe."), n_init_dd], run_plots,"OK (Daten vorbereiten & Ellbogen-Plots generieren)" )        

    def _execute_preparation_and_plots(self):
        self.app = ClusteringApp(
            original_df=self.sampled_df, 
            comparison_column=self.comparison_col, 
            treatment_map=self.treatment_map, 
            elbow_style=self.elbow_style, 
            n_init=self.n_init,
            cat_cleaning_map=self.cat_cleaning_map
        )
        if self.app.out not in self.all_steps_container.children:
            self.all_steps_container.children += (self.app.out,)
        if self.app.prepare_data():
            if self.app.show_elbow():
                self.setup_k_selection()
    def setup_k_selection(self):
        if self.app.X_scaled is None or len(self.app.X_scaled) < 2: return
        k_options = list(range(2, self.app.max_k)) 
        if not k_options: return
        default_k = min(4, k_options[-1]) 
        k_dd = Dropdown(options=k_options, value=default_k, description="Musterzahl k:")
        def run(b, output_for_step):
            k = k_dd.value
            with output_for_step:
                clear_output() 
                print(f"✅ Auswahl: k={k}. Starte Methodenvergleich...")
            k_dd.disabled = True
            b.disabled = True
            self.app.compare_methods(k)
            self.best_labels = self.app.show_visualization_controls()  
            self.setup_comparison_selection()
        self._add_step_widgets("5️⃣ Musterzahl (k) auswählen:",[Label("Basierend auf den Plots (Schritt 2), wählen Sie die gewünschte Musterzahl 'k'."), k_dd], run, "Clustering Methoden vergleichen & Visualisierung starten")
    
    # SCHRITT 6: LÖSUNGSSPALTE WÄHLEN - KORRIGIERT
    def setup_comparison_selection(self):
        all_cols = self.selected_df.columns.tolist()
        comparison_options = ["Kein Vergleich"] + all_cols
        solution_dd = Dropdown(options=comparison_options, 
                               value="Kein Vergleich", 
                               description="Lösungsspalte (Validation Column):", 
                               style={'description_width': 'initial'})
        def run_comparison(b, output_for_step):
            self.solution_col_name = solution_dd.value
            with output_for_step:
                clear_output()
                if self.app.selected_method_from_viz is None:
                    print("❌ FEHLER: Es wurde keine Methode zur Zuordnung gefunden. Führen Sie **Schritt 5** erneut aus.")
                    return
                print(f"✅ Lösungsspalte gewählt: **'{self.solution_col_name}'**.")
                
                solution_dd.disabled = True
                b.disabled = True
                
                # KORREKTUR: Wenn "Kein Vergleich" gewählt ist, überspringe den Vergleichsschritt (implizit in Schritt 7)
                if self.solution_col_name == "Kein Vergleich":
                    print("➡️ **Kein Vergleich** ausgewählt (Szenario A). Fahren Sie mit **Schritt 7** fort, um nur die Muster zuzuordnen.")
                else:
                    print("➡️ Vergleichsspalte ausgewählt (Szenario B). Fahren Sie mit **Schritt 7** fort, um die Muster zuzuordnen, den Vergleich und die **Merkmalswichtigkeit** durchzuführen.")
                    
                self.setup_final_assignment()
                
        self._add_step_widgets("6️⃣ Lösungsspalte (Validation Column) wählen:",[Label("Diese Spalte wird für den Muster-Vergleich und die Generierung der Spalte **'TEMP_Clear_ML'** in df_ML verwendet."), solution_dd], run_comparison,"OK (Vergleichs-Spalte speichern)",btn_style='info')

    # SCHRITT 7: MUSTER-ZUORDNUNG (Cluster_ML) UND VERGLEICH (TEMP_Clear_ML) DURCHFÜHREN
    def setup_final_assignment(self):
        def assign_musters_and_compare(b, output_for_step):
            with output_for_step:
                clear_output()
                global df_ML
                if 'df_ML' not in globals() or globals()['df_ML'].empty:
                     df_ML = self.sampled_df.copy()
                     globals()['df_ML'] = df_ML
                     # BEREINIGT: Entfernen von '\'
                     print("ℹ️ df_ML wurde aus dem Sample-DataFrame re-initialisiert.")
                if self.app is None or self.app.selected_method_from_viz is None:
                    print("❌ FEHLER: ClusteringApp wurde nicht richtig initialisiert oder Schritt 5 wurde nicht ausgeführt.")
                    return
                
                self.app.execute_assignment_and_comparison(self.solution_col_name, df_ML, self.app.selected_method_from_viz)
                
                # BEREINIGT: Entfernen von '\'
                print("\n✅ Schritt 7 abgeschlossen. df_ML enthält nun die Zuordnung(en).")
                print("➡️ Fahren Sie mit **Schritt 8** fort, um die finalen Spalten zuzuordnen und df_ML zu löschen.")
            b.disabled = True
            self.setup_final_cleanup()
        
        # Logik, die den Titel für den Button anpasst
        if self.solution_col_name and self.solution_col_name != "Kein Vergleich":
            # BEREINIGT: Entfernen von '\'
            label_text = "Dieser Schritt fügt **Cluster_ML** und **TEMP_Clear_ML** (Vergleich) in den Arbeits-DataFrame **df_ML** ein und generiert die **Merkmalswichtigkeit**."
        else:
            # BEREINIGT: Entfernen von '\'
            label_text = "Dieser Schritt fügt **Cluster_ML** in den Arbeits-DataFrame **df_ML** ein und generiert die finale deskriptive Cluster-Analyse (Kein Vergleich und keine Merkmalswichtigkeit)."

        self._add_step_widgets(
            # BEREINIGT: Entfernen von '\'
            "7️⃣ Cluster-Zuordnung (Cluster_ML) & Vergleich (TEMP_Clear_ML) durchführen:",
            [Label(f"Cluster-Zuordnung erfolgt mit der in **Schritt 4** gewählten Methode: **{self.app.selected_method_from_viz if self.app and self.app.selected_method_from_viz else 'Nicht gewählt/Beste'}**."), 
             Label(label_text), 
             Label("Er generiert auch die finale deskriptive Cluster-Analyse.")],
            assign_musters_and_compare,
            "OK (Muster zuordnen & Analyse generieren)",
            btn_style='warning'
        )
    # SCHRITT 8: FINALE SPEICHERUNG UND BEREINIGUNG
    def setup_final_cleanup(self): 
        assignment_options = { " Muster in DataFrame einbinden (Spalte: 'Cluster_ML')": "assign", "Keine (Analyse abgeschlossen)": "none"}
        assign_dd = Dropdown(options=assignment_options, value="assign", description="Zuordnung:", style={'description_width': 'initial'})
        def final_cleanup(b, output_for_step):
            with output_for_step:
                clear_output()
                global df_ML 
                target_df_name = self.selected_df_name
                
                if assign_dd.value == "assign":
                    # BEREINIGT: Entfernen von '\'
                    if 'df_ML' not in globals() or globals()['df_ML'].empty or 'Cluster_ML' not in globals()['df_ML'].columns:
                        print("❌ FEHLER: df_ML enthält keine 'Cluster_ML' Spalte. Führen Sie **Schritt 7** aus.")
                        return
                    # 1. Zuweisung an das globale, vom Benutzer gewählte DataFrame (Weiterarbeiten)
                    if target_df_name in globals() and isinstance(globals()[target_df_name], pd.DataFrame):
                        df_target = globals()[target_df_name]
                        # Sicherstellen, dass die Spalten erstellt werden (falls sie im Original-DF fehlen)
                        df_target['Cluster_ML'] = pd.Series(pd.NA, index=df_target.index, dtype='Int64') 
                        if 'TEMP_Clear_ML' not in df_target.columns:
                            df_target['TEMP_Clear_ML'] = pd.Series(pd.NA, index=df_target.index, dtype=object)
                        # Werte aus df_ML übertragen
                        df_target['Cluster_ML'].update(df_ML['Cluster_ML'])
                        if 'TEMP_Clear_ML' in df_ML.columns:
                            df_target['TEMP_Clear_ML'].update(df_ML['TEMP_Clear_ML'])
                        globals()[target_df_name] = df_target
                        target_name_display = f"**{target_df_name}**"
                        # BEREINIGT: Entfernen von '\'
                        print(f"✅ Finale Spalten **'Cluster_ML'** und optional **'TEMP_Clear_ML'** in {target_name_display} **aktualisiert**.")
                    # 2. Lösche df_ML nach der finalen Zuweisung
                    if 'df_ML' in globals():
                        del globals()['df_ML']
                        print("\n🗑️ Der Arbeits-DataFrame **df_ML** wurde aus dem globalen Bereich entfernt (Aufgabe vollendet).")
                else:
                    # BEREINIGT: Entfernen von '\'
                    print("✅ Analyse abgeschlossen. Die DataFrames wurden nicht verändert (df_ML bleibt ungelöscht).")
                assign_dd.disabled = True
                b.disabled = True
        # BEREINIGT: Entfernen von '\'
        self._add_step_widgets("8️⃣ Finale Speicherung & Bereinigung (Löscht df_ML):",
                               [Label("Überträgt die Cluster-Ergebnisse in das Ausgangs-DataFrame und löscht den temporären Arbeits-DataFrame."), assign_dd],
                               final_cleanup,
                               "OK (Finale Speicherung)",
                               btn_style='success')
# Führen Sie das interaktive Clustering-Tool aus
ClusteringTool()