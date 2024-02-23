import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


lgbm_res = {'training_perf': {'MSE': [246319.03234714657, 233750.38516247307, 227264.05317319283, 220909.7400029852, 228654.62543681005, 241148.0357067766, 236832.94336951643, 217408.15752177738, 218228.53687859667, 222390.003118489], 'MAE': [325.5092192366293, 321.58211030251124, 316.4661909368421, 312.3210291774034, 315.5790090758659, 323.4046312082966, 324.5864273656323, 308.84759782186563, 309.047820478118, 319.49193244479306], 'R2': [0.8583199533404477, 0.8683577199552737, 0.87059341972369, 0.8736712639872086, 0.874690952859247, 0.8639675238103282, 0.870384116536497, 0.8780770068236092, 0.8760122071426598, 0.8792293086452614], 'MSE_avg': 229290.55127177638, 'MAE_avg': 317.6835968047958, 'R2_avg': 0.8713303472824222, 'MSE_std': 9451.312099196479,'MAE_std': 5.894134075733489,'R2_std': 0.006167339996084232}, 'test_perf': {'MSE': 224600.67783899326, 'MAE': 316.58123215634424, 'R2': 0.8745924178933205}, 'params': {'boosting_type': 'dart', 'learning_rate': 0.1, 'n_estimators': 3200, 'importance_type': 'gain'}}
svr_res = {'training_perf': {'MSE': [288593.36834768567, 277839.2610063708, 267128.69156774617, 266527.7792293521, 263338.47932938294, 290026.90736972896, 286771.4823402629, 256470.62049742232, 268928.2503125384, 277350.42835927923], 'MAE': [341.1932549109759, 337.3144961448437, 333.9971229683854, 331.3397339651103, 329.828750810093, 339.7068263511721, 342.91251628966216, 326.2189117016068, 330.48583238729026, 337.7156172902185], 'R2': [0.8340042119217463, 0.8435279848655736, 0.847894068653596, 0.8475842782582362, 0.855683243419105, 0.8363947761154078, 0.8430533417064485, 0.8561706880307851, 0.8472068746362555, 0.8493826048348201],'MSE_avg': 274297.526835977, 'MAE_avg': 335.07130628193585, 'R2_avg': 0.8460902072441975, 'MSE_std': 10989.71232692334, 'MAE_std': 5.238601983833233,'R2_std': 0.00683429317193578}, 'test_perf': {'MSE': 264313.3503861633, 'MAE': 334.425020644407,'R2': 0.8524185300357556}, 'params': {'C': 4096, 'kernel': 'rbf', 'epsilon': 0.1}}

res = stats.wilcoxon(lgbm_res['training_perf']['R2'], 
                     svr_res['training_perf']['R2'],
                     alternative='greater')

print('R2: ' + str(res.statistic))
print('p: ' + str(res.pvalue/2))


df = pd.DataFrame(
                  {
                   'LGBM': lgbm_res['training_perf']['R2'], 
                   'SVR': svr_res['training_perf']['R2'],
                  }
                 )
data = df.values

labels = np.array(list(df.keys()))

# Change colors and fill boxes
plt.boxplot(data, 
            labels=labels,
            patch_artist=True, 
            boxprops=dict(facecolor='lightblue'), 
            medianprops=dict(linewidth=2, color='red'))
# plt.ylim(bottom=0, top=1)

# Show the plot
plt.tight_layout()
# plt.show()

res = stats.wilcoxon(lgbm_res['training_perf']['MSE'], 
                     svr_res['training_perf']['MSE'],
                     alternative='less')

print('MSE: ' + str(res.statistic))
print('p: ' + str(res.pvalue/2))

res = stats.wilcoxon(lgbm_res['training_perf']['MAE'], 
                     svr_res['training_perf']['MAE'],
                     alternative='less')

print('MAE: ' + str(res.statistic))
print('p: ' + str(res.pvalue/2))
