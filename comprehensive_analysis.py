import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, brier_score_loss
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OBJEXAnalyzer:
    def __init__(self, excel_path: str):
        """Initialize analyzer with Excel file path"""
        self.excel_path = excel_path
        self.data = {}
        self.extracted_data = {}
        self.similarity_data = {}
        self.models = ['gpt-4.1', 'claude-sonnet-4', 'Qwen3-235B-A22B-fp8-',
                      'moonshotaiKimi-K2-In', 'deepseek-aiDeepSeek-', 'gemini-2.5-flash']
        self.datasets = ['SafeMTData_Attack600', 'SafeMTData_1K', 'MHJ', 'CoSafe']
        self.optimal_threshold = None

    def load_data(self):
        """Load all sheets from Excel file"""
        print("Loading data from Excel file...")
        xl_file = pd.ExcelFile(self.excel_path)

        for sheet_name in xl_file.sheet_names:
            print(f"  Loading sheet: {sheet_name}")
            df = pd.read_excel(xl_file, sheet_name=sheet_name)
            print(f"    Shape: {df.shape}")

            # Organize data by type
            if sheet_name.startswith('extracted_'):
                model_name = sheet_name.replace('extracted_', '')
                self.extracted_data[model_name] = df
            elif sheet_name.startswith('similarity_'):
                model_name = sheet_name.replace('similarity_', '')
                self.similarity_data[model_name] = df
            else:
                self.data[sheet_name] = df

        # Load labeling sheet for human alignment
        if 'Labeling' in self.data:
            self.labeling_data = self.data['Labeling']
            print(f"\nLabeling data loaded: {len(self.labeling_data)} samples")

        # Merge extracted and similarity data for each model
        self.merged_data = {}
        for model in self.models:
            # Find matching keys
            extract_key = None
            sim_key = None

            for key in self.extracted_data.keys():
                if key.startswith(model):
                    extract_key = key
                    break

            for key in self.similarity_data.keys():
                if key.startswith(model):
                    sim_key = key
                    break

            if extract_key and sim_key:
                # Merge on base_prompt
                merged = pd.merge(
                    self.extracted_data[extract_key][['source', 'base_prompt', 'extracted_base_prompt', 'extraction_confidence']],
                    self.similarity_data[sim_key][['base_prompt', 'similarity_score', 'similarity_category']],
                    on='base_prompt',
                    how='inner'
                )
                merged.rename(columns={'extraction_confidence': 'confidence'}, inplace=True)
                self.merged_data[model] = merged
                print(f"  Merged data for {model}: {len(merged)} samples")

        return self.data

    def human_aligned_thresholding(self, similarity_col='similarity_score',
                                  human_col='human_label'):
        """Find optimal threshold using human labels"""
        print("\n=== Human-aligned Thresholding Analysis ===")

        if not hasattr(self, 'labeling_data'):
            print("Error: Labeling sheet not found")
            return None, None

        df = self.labeling_data.copy()

        # Convert human labels to binary
        # Exact match + High similarity -> 1, Moderate + Low -> 0
        label_mapping = {
            'Exact match': 1,
            'High similarity': 1,
            'Moderate similarity': 0,
            'Low similarity': 0
        }

        df['human_binary'] = df[human_col].map(label_mapping)

        # Check for NaN values
        if df['human_binary'].isna().any():
            print("Warning: Some human labels could not be mapped")
            df = df.dropna(subset=['human_binary'])

        # Find optimal threshold by F1 score
        thresholds = np.arange(0.0, 1.01, 0.01)
        best_f1 = 0
        best_threshold = 0

        results = []
        for threshold in thresholds:
            predictions = (df[similarity_col] >= threshold).astype(int)
            f1 = f1_score(df['human_binary'], predictions)
            acc = accuracy_score(df['human_binary'], predictions)

            results.append({
                'threshold': threshold,
                'f1': f1,
                'accuracy': acc
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.optimal_threshold = best_threshold

        print(f"Optimal threshold: {best_threshold:.2f}")
        print(f"Best F1 score: {best_f1:.3f}")

        # Calculate metrics at optimal threshold
        predictions = (df[similarity_col] >= best_threshold).astype(int)

        from sklearn.metrics import classification_report
        print("\nClassification Report at Optimal Threshold:")
        print(classification_report(df['human_binary'], predictions))

        return pd.DataFrame(results), best_threshold

    def calculate_accuracy_with_bootstrap(self, model_data: pd.DataFrame,
                                         threshold: float = 0.58,
                                         n_bootstrap: int = 10000) -> Dict:
        """Calculate accuracy with bootstrap confidence intervals"""

        # Apply threshold to get binary predictions
        predictions = (model_data['similarity_score'] >= threshold).astype(int)

        # For accuracy calculation, we consider prediction=1 as correct extraction
        # Base accuracy is simply the mean of predictions
        base_accuracy = predictions.mean()

        # Bootstrap for CI
        accuracies = []
        n_samples = len(predictions)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_preds = predictions.iloc[indices]
            accuracies.append(boot_preds.mean())

        ci_lower = np.percentile(accuracies, 2.5)
        ci_upper = np.percentile(accuracies, 97.5)

        return {
            'accuracy': base_accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(accuracies)
        }

    def dataset_wise_performance(self, threshold: float = None):
        """Analyze performance across datasets"""
        print("\n=== Dataset-wise Performance Analysis ===")

        if threshold is None:
            threshold = self.optimal_threshold if self.optimal_threshold else 0.58

        results = []

        for model in self.models:
            if model not in self.merged_data:
                continue

            model_df = self.merged_data[model]

            for dataset in self.datasets:
                dataset_data = model_df[model_df['source'] == dataset]

                if len(dataset_data) == 0:
                    continue

                metrics = self.calculate_accuracy_with_bootstrap(
                    dataset_data, threshold, n_bootstrap=1000
                )

                results.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Accuracy': metrics['accuracy'],
                    'CI_Lower': metrics['ci_lower'],
                    'CI_Upper': metrics['ci_upper'],
                    'N_Samples': len(dataset_data)
                })

        results_df = pd.DataFrame(results)

        # Calculate performance spread
        print("\nPerformance Spread (max - min):")
        for model in self.models:
            model_data = results_df[results_df['Model'] == model]
            if len(model_data) > 0:
                spread = model_data['Accuracy'].max() - model_data['Accuracy'].min()
                print(f"  {model}: {spread:.3f}")

        return results_df

    def pairwise_model_comparison(self, threshold: float = None, n_bootstrap: int = 10000):
        """Perform pairwise comparison between models"""
        print("\n=== Pairwise Model Comparison ===")

        if threshold is None:
            threshold = self.optimal_threshold if self.optimal_threshold else 0.58

        comparisons = []

        model_list = list(self.merged_data.keys())

        for i, model1 in enumerate(model_list):
            for model2 in model_list[i+1:]:
                df1 = self.merged_data[model1]
                df2 = self.merged_data[model2]

                # Align data by base_prompt
                merged = pd.merge(
                    df1[['base_prompt', 'similarity_score', 'confidence']],
                    df2[['base_prompt', 'similarity_score', 'confidence']],
                    on='base_prompt',
                    suffixes=('_1', '_2')
                )

                if len(merged) == 0:
                    continue

                # Calculate differences
                pred1 = (merged['similarity_score_1'] >= threshold).astype(int)
                pred2 = (merged['similarity_score_2'] >= threshold).astype(int)

                diff = pred1.mean() - pred2.mean()

                # Bootstrap for significance
                diffs = []
                n_samples = len(merged)

                for _ in range(n_bootstrap):
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    boot_pred1 = pred1.iloc[indices]
                    boot_pred2 = pred2.iloc[indices]
                    diffs.append(boot_pred1.mean() - boot_pred2.mean())

                ci_lower = np.percentile(diffs, 2.5)
                ci_upper = np.percentile(diffs, 97.5)

                # Check significance (CI doesn't contain 0)
                significant = (ci_lower > 0) or (ci_upper < 0)

                comparisons.append({
                    'Model1': model1,
                    'Model2': model2,
                    'Difference': diff,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'Significant': significant,
                    'P_value': 2 * min(sum(np.array(diffs) <= 0) / n_bootstrap,
                                      sum(np.array(diffs) >= 0) / n_bootstrap)
                })

        return pd.DataFrame(comparisons)

    def calculate_metacognition_metrics(self, model_data: pd.DataFrame,
                                       threshold: float = None) -> Dict:
        """Calculate ECE, Brier score, Wrong@High-Conf, and AURC"""

        if threshold is None:
            threshold = self.optimal_threshold if self.optimal_threshold else 0.58

        # Binary predictions
        predictions = (model_data['similarity_score'] >= threshold).astype(int)
        confidences = model_data['confidence'].values / 100.0  # Convert to 0-1 scale if needed

        # Clip confidences to [0, 1]
        confidences = np.clip(confidences, 0, 1)

        # ECE (Expected Calibration Error)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            if i == n_bins - 1:  # Include 1.0 in the last bin
                bin_mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])

            if bin_mask.sum() > 0:
                bin_conf = confidences[bin_mask].mean()
                bin_acc = predictions[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(predictions)
                ece += bin_weight * abs(bin_conf - bin_acc)

        # Brier Score
        brier = np.mean((confidences - predictions)**2)

        # Wrong@High-Confidence
        wrong_at_80 = ((predictions == 0) & (confidences >= 0.8)).mean()
        wrong_at_90 = ((predictions == 0) & (confidences >= 0.9)).mean()
        wrong_at_95 = ((predictions == 0) & (confidences >= 0.95)).mean()

        # AURC (Area Under Risk-Coverage curve)
        sorted_indices = np.argsort(-confidences)  # Sort by confidence descending
        sorted_preds = predictions.iloc[sorted_indices].values
        sorted_conf = confidences[sorted_indices]

        coverages = np.arange(1, len(predictions) + 1) / len(predictions)
        risks = 1 - np.cumsum(sorted_preds) / np.arange(1, len(predictions) + 1)
        aurc = np.trapz(risks, coverages)

        return {
            'ECE': ece,
            'Brier': brier,
            'Wrong@0.8': wrong_at_80,
            'Wrong@0.9': wrong_at_90,
            'Wrong@0.95': wrong_at_95,
            'AURC': aurc
        }

    def run_complete_analysis(self):
        """Run all analyses and generate comprehensive report"""
        print("="*60)
        print("OBJEX Dataset Comprehensive Analysis")
        print("="*60)

        # Load data
        self.load_data()

        # 1. Human-aligned thresholding
        threshold_results, optimal_threshold = self.human_aligned_thresholding()

        # 2. Overall model accuracy
        print("\n=== Overall Model Accuracy ===")
        model_accuracies = []

        for model in self.merged_data.keys():
            metrics = self.calculate_accuracy_with_bootstrap(
                self.merged_data[model], optimal_threshold
            )

            model_accuracies.append({
                'Model': model,
                'Accuracy': metrics['accuracy'],
                'CI_Lower': metrics['ci_lower'],
                'CI_Upper': metrics['ci_upper'],
                'N_Samples': len(self.merged_data[model])
            })

            print(f"{model}: {metrics['accuracy']:.3f} [{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}]")

        # 3. Dataset-wise performance
        dataset_results = self.dataset_wise_performance(optimal_threshold)

        # 4. Pairwise comparisons
        pairwise_results = self.pairwise_model_comparison(optimal_threshold)

        # 5. Metacognition metrics
        print("\n=== Metacognition Analysis ===")
        meta_results = []

        for model in self.merged_data.keys():
            metrics = self.calculate_metacognition_metrics(
                self.merged_data[model], optimal_threshold
            )

            meta_results.append({
                'Model': model,
                **metrics
            })

            print(f"\n{model}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        # 6. Per-source detailed metrics
        print("\n=== Per-source Detailed Metrics ===")
        source_metrics = []

        for model in self.merged_data.keys():
            model_df = self.merged_data[model]

            for source in self.datasets:
                source_data = model_df[model_df['source'] == source]
                if len(source_data) == 0:
                    continue

                coverage = len(source_data) / len(model_df[model_df['source'] == source])
                error_rate = 1 - (source_data['similarity_score'] >= optimal_threshold).mean()

                source_metrics.append({
                    'Model': model,
                    'Source': source,
                    'Coverage': coverage,
                    'Error_Rate': error_rate,
                    'Mean_Similarity': source_data['similarity_score'].mean(),
                    'Mean_Confidence': source_data['confidence'].mean()
                })

        source_metrics_df = pd.DataFrame(source_metrics)

        # Save all results
        results = {
            'threshold_optimization': threshold_results,
            'model_accuracies': pd.DataFrame(model_accuracies),
            'dataset_performance': dataset_results,
            'pairwise_comparisons': pairwise_results,
            'metacognition': pd.DataFrame(meta_results),
            'source_metrics': source_metrics_df
        }

        return results

    def create_visualizations(self, results: Dict):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # 1. Threshold optimization curve
        ax = axes[0, 0]
        if results['threshold_optimization'] is not None:
            threshold_df = results['threshold_optimization']
            ax.plot(threshold_df['threshold'], threshold_df['f1'], label='F1 Score', linewidth=2)
            ax.plot(threshold_df['threshold'], threshold_df['accuracy'], label='Accuracy', linewidth=2)
            ax.axvline(x=self.optimal_threshold, color='r', linestyle='--', label=f'Optimal: {self.optimal_threshold:.2f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title('Threshold Optimization')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Model accuracy comparison
        ax = axes[0, 1]
        model_acc = results['model_accuracies'].sort_values('Accuracy', ascending=False)
        bars = ax.bar(range(len(model_acc)), model_acc['Accuracy'])
        ax.errorbar(range(len(model_acc)), model_acc['Accuracy'],
                   yerr=[model_acc['Accuracy'] - model_acc['CI_Lower'],
                         model_acc['CI_Upper'] - model_acc['Accuracy']],
                   fmt='none', color='black', capsize=5)
        ax.set_xticks(range(len(model_acc)))
        ax.set_xticklabels(model_acc['Model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance (95% CI)')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Dataset-wise heatmap
        ax = axes[0, 2]
        if len(results['dataset_performance']) > 0:
            pivot = results['dataset_performance'].pivot(index='Model', columns='Dataset', values='Accuracy')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
            ax.set_title('Performance by Dataset')

        # 4. ECE comparison
        ax = axes[1, 0]
        meta_df = results['metacognition'].sort_values('ECE')
        ax.barh(range(len(meta_df)), meta_df['ECE'])
        ax.set_yticks(range(len(meta_df)))
        ax.set_yticklabels(meta_df['Model'])
        ax.set_xlabel('Expected Calibration Error')
        ax.set_title('Calibration Quality (lower is better)')
        ax.grid(True, alpha=0.3, axis='x')

        # 5. Wrong@High-Confidence
        ax = axes[1, 1]
        x = np.arange(len(meta_df))
        width = 0.25
        ax.bar(x - width, meta_df['Wrong@0.8'], width, label='@0.8')
        ax.bar(x, meta_df['Wrong@0.9'], width, label='@0.9')
        ax.bar(x + width, meta_df['Wrong@0.95'], width, label='@0.95')
        ax.set_xticks(x)
        ax.set_xticklabels(meta_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('Error Rate')
        ax.set_title('Errors at High Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Pairwise significance matrix
        ax = axes[1, 2]
        if len(results['pairwise_comparisons']) > 0:
            # Create significance matrix
            models = list(self.merged_data.keys())
            sig_matrix = np.zeros((len(models), len(models)))

            for _, row in results['pairwise_comparisons'].iterrows():
                try:
                    i = models.index(row['Model1'])
                    j = models.index(row['Model2'])
                    if row['Significant']:
                        sig_matrix[i, j] = row['Difference']
                        sig_matrix[j, i] = -row['Difference']
                except ValueError:
                    continue

            sns.heatmap(sig_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, ax=ax, xticklabels=models, yticklabels=models)
            ax.set_title('Pairwise Differences (significant only)')

        # 7. Brier Score comparison
        ax = axes[2, 0]
        meta_df_sorted = results['metacognition'].sort_values('Brier')
        ax.barh(range(len(meta_df_sorted)), meta_df_sorted['Brier'])
        ax.set_yticks(range(len(meta_df_sorted)))
        ax.set_yticklabels(meta_df_sorted['Model'])
        ax.set_xlabel('Brier Score')
        ax.set_title('Prediction Reliability (lower is better)')
        ax.grid(True, alpha=0.3, axis='x')

        # 8. AURC comparison
        ax = axes[2, 1]
        meta_df_aurc = results['metacognition'].sort_values('AURC')
        ax.barh(range(len(meta_df_aurc)), meta_df_aurc['AURC'])
        ax.set_yticks(range(len(meta_df_aurc)))
        ax.set_yticklabels(meta_df_aurc['Model'])
        ax.set_xlabel('AURC')
        ax.set_title('Risk-Coverage Trade-off (lower is better)')
        ax.grid(True, alpha=0.3, axis='x')

        # 9. Sample sizes by dataset
        ax = axes[2, 2]
        if len(results['dataset_performance']) > 0:
            sample_pivot = results['dataset_performance'].pivot(index='Model', columns='Dataset', values='N_Samples')
            sns.heatmap(sample_pivot, annot=True, fmt='g', cmap='Blues', ax=ax)
            ax.set_title('Sample Sizes by Dataset')

        plt.tight_layout()
        plt.savefig('objex_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def generate_latex_tables(self, results: Dict):
        """Generate LaTeX tables for paper"""

        # Main results table
        main_table = results['model_accuracies'].copy()
        main_table['Accuracy_CI'] = main_table.apply(
            lambda x: f"{x['Accuracy']:.3f} [{x['CI_Lower']:.3f}, {x['CI_Upper']:.3f}]", axis=1
        )

        # Metacognition table
        meta_table = results['metacognition'].copy()
        meta_table = meta_table.round(4)

        # Save to LaTeX
        with open('latex_tables.tex', 'w') as f:
            f.write("% Main Results Table\n")
            f.write(main_table[['Model', 'Accuracy_CI', 'N_Samples']].to_latex(index=False))
            f.write("\n\n% Metacognition Metrics Table\n")
            f.write(meta_table.to_latex(index=False))

        print("\nLaTeX tables saved to latex_tables.tex")

def main():
    # Initialize analyzer
    analyzer = OBJEXAnalyzer('E:\\Project\\OBJEX_dataset\\OBJEX_dataset_labeling.xlsx')

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    # Create visualizations
    analyzer.create_visualizations(results)

    # Generate LaTeX tables
    analyzer.generate_latex_tables(results)

    # Save results to Excel
    with pd.ExcelWriter('objex_analysis_output.xlsx', engine='openpyxl') as writer:
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sheet_name = name[:30]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save results to pickle for later use
    import pickle
    with open('objex_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*60)
    print("Analysis complete! Results saved to:")
    print("  - objex_analysis_output.xlsx (numerical results)")
    print("  - objex_analysis_results.png (visualizations)")
    print("  - objex_analysis_results.pkl (Python pickle)")
    print("  - latex_tables.tex (LaTeX formatted tables)")
    print("="*60)

if __name__ == "__main__":
    main()