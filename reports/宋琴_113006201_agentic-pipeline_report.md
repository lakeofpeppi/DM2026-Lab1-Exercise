# AgenticPipeline Analysis Report

## 1. Dataset Loading and Exploration

I analyzed the 20 Newsgroups dataset using four categories: alt.atheism, comp.graphics, sci.med, and soc.religion.christian. The dataset contained 2,257 documents, with category counts of 480, 584, 594, and 599, respectively (`load_dataset_1`). Using `inspect_data_tool`, `check_missing_tool`, and `check_duplicates_tool`, I verified that the dataset contained three columns (text, category, and category_name), with no missing values or duplicate rows.

The descriptive statistics (`describe_data_5`) showed substantial variation in document length. The average document contained approximately 1,993 characters, while the longest contained 60,713 characters. The large outliers could disproportionately influence raw word-frequency analysis and dimensionality reduction.

![Document length distribution](../plots/宋琴_agentic-pipeline_plot_describe_data_5.png)

## 2. Document-Term Matrix and Term Frequency

I constructed a baseline document-term matrix (`dtm_6`) using 2,257 documents and 35,788 unique terms. The resulting matrix contained 365,886 non-zero entries and had a sparsity of approximately 99.55%. This indicates that most words appear in only a small proportion of documents.

Term-frequency analysis (`term_freq_7`) showed that the most common terms included "the" (29,825 occurrences), "of" (17,660), and "to" (17,222). These terms carry relatively little information for distinguishing topics. The heatmap (`dtm_heatmap_8`) also demonstrated the sparsity of the matrix, although its first 20 alphabetically sorted terms were mostly numeric tokens.

![Document-term matrix heatmap](../plots/宋琴_agentic-pipeline_plot_dtm_heatmap_8.png)

## 3. Feature Selection and Correlation Analysis

Variance filtering (`variance_filter_9`) with a threshold of 0.01 reduced the vocabulary from 35,788 to 5,354 terms, an 85.04% reduction. Pearson filtering with a threshold of 0.05 (`pearson_filter_11`) retained 2,427 terms, while Spearman filtering at the same threshold (`spearman_filter_12`) retained 2,779 terms. Both correlation experiments focused on the soc.religion.christian category.

These results demonstrate how different feature-selection methods produce substantially different retained vocabularies. However, feature reduction alone does not establish which method produces better classification performance.

The feature correlation matrix (`feature_corr_13`) showed strong positive relationships between common words, such as "the" and "of" (approximately 0.94). This suggests considerable redundancy among frequent terms, although differences in document length may also contribute to the correlations. The matrix tool evaluated the top 20 global terms rather than the Spearman-filtered feature set.

![Feature correlation matrix](../plots/宋琴_agentic-pipeline_plot_feature_corr_13.png)

## 4. Dimensionality Reduction

For dimensionality reduction, I rebuilt a filtered DTM (`dtm_2`) using min_df=20, reducing the vocabulary to 2,682 terms while keeping all 2,257 documents.

PCA (`reduce_pca_3`) reduced this matrix to two dimensions. The first principal component explained 69.76% of the variance, and the second explained 4.40%, for a combined explained variance of 74.16%.

Despite capturing much of the variance, the scatter plot showed substantial overlap among the four categories. Several extreme outliers stretched the PCA axes and compressed most documents into a dense cluster. This suggests that maximizing variance does not necessarily produce useful separation between document categories.

![PCA dimensionality reduction](../plots/宋琴_agentic-pipeline_plot_reduce_pca_3.png)

## 5. Label Binarization and Cosine Similarity

Label binarization (`binarize_labels_4`) converted the four categories into a binary indicator matrix with dimensions 2,257 × 4. This representation can support one-vs-rest classification and category-specific feature analysis.

Cosine similarity (`cosine_sim_5`) between documents 0 and 1 returned a score of 0.4387. Both documents were verified as belonging to comp.graphics. Their moderate similarity shows that documents within the same category can still differ considerably in vocabulary and content.

## 6. Pattern Mining and Limitations

I performed frequent pattern mining on the soc.religion.christian category using   mine_patterns_tool (mine_patterns_7). The experiment used the FP-Growth algorithm with term-frequency filtering and a minimum support threshold of 500. No frequent patterns were discovered under these settings. Since the category contains 599 documents, the minimum support requirement corresponds to approximately 83.5% of the category. This is a highly restrictive threshold for text data, where individual terms and combinations of terms may not occur consistently across documents. The result does not imply that meaningful frequent patterns are absent, only that none satisfied this threshold. An earlier attempt using the default minimum support of 1 resulted in an excessively long computation. The experiment completed successfully with a higher minimum support threshold, suggesting that more restrictive settings can help manage the computational cost of frequent pattern mining.

One additional limitation was encountered when `visualize_result_tool` could not visualize `term_freq_7` because the stored result was not in the expected tabular format. The term-frequency values were still available for interpretation.

Overall, the analysis demonstrated the importance of preprocessing, feature selection, and careful interpretation when working with sparse, high-dimensional text data.