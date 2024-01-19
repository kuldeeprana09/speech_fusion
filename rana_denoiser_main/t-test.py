import scipy.stats as stats

# Assuming you have two arrays of performance metrics for the two models
# model1_metrics = [1.2, 1.5, 1.8, 1.4, 1.6]
# model2_metrics = [1.1, 1.3, 1.7, 1.2, 1.5]

TCN_SL1_SNRi = [6.01,8.17,7.10,9.72,5.53]
# TCN_SL1_SNRi = [8.07,5.76,7.72,6.01,8.17,7.10,9.72,5.53,7.31]
# TCN_SL2_SNRi = [7.22,4.97,7.11,5.62,7.75,6.41,9.11,4.61,6.70]

TCN1_FT_SNRi = [5.97,8.05,7.16,9.40,6.44]
# TCN1_FT_SNRi = [7.28,6.06,7.05,5.97,8.05,7.16,9.40,6.44,7.40]
# TCN2_FT_SNRi = [9.21,6.42,8.63,7.29,8.88,8.16,10.87,6.19,8.28]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(TCN_SL1_SNRi, TCN1_FT_SNRi)

# Print the t-statistic and p-value
print("t-statistic:", t_statistic)
print("p-value:", p_value)