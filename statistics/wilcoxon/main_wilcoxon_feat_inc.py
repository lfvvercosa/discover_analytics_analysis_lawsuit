import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


reg = {'training_perf': {'MSE': [439760.3793652453, 432866.71454576607, 404593.64694551827, 417785.8408093249, 423327.98346281867, 437586.28033570613, 448444.73779895966, 404587.69639660686, 402625.95057367854, 422451.410700933], 'MAE': [464.4886637777006, 462.766572056304, 448.12492338952393, 455.7597265005221, 454.8399810999965, 466.87412613297965, 470.47111131523985, 449.68789942973075, 445.78220547858086, 461.4684680545499], 'R2': [0.7470545800956168, 0.7562204604768169, 0.7696200541981884, 0.7610863278696082, 0.7680045783705257, 0.7531560019295597, 0.7545714711501043, 0.7731062922877031, 0.7712457606472867, 0.770584341837735], 'MSE_avg': 423403.0640934557, 'MAE_avg': 458.0263677235128, 'R2_avg': 0.7624649868863145, 'MSE_std': 15353.620575168683, 'MAE_std': 8.01256649674511, 'R2_std': 0.008749203364789715},'test_perf': {'MSE': 407939.45752992365, 'MAE': 449.37710074130604, 'R2': 0.7722237461304027}, 'params': {'boosting_type': 'dart', 'learning_rate': 0.2, 'n_estimators': 600, 'importance_type': 'split'}}
reg_cus = {'training_perf': {'MSE': [310199.7770259049, 286590.9682020449, 283545.10366714455, 281616.78214592504, 297436.1053784361, 304397.3910678903, 310794.15405720356, 281620.3257168014, 281322.29299308255, 290873.86674429197], 'MAE': [371.4559478364773, 363.50399891854914, 362.2277704942583, 361.1488417410527, 365.8303201618033, 373.3709913809643, 375.9294853633283, 359.6700269605202, 359.09002880066544, 372.78854683240434], 'R2': [0.8215764390431926, 0.8385992456520677, 0.8385463881888335, 0.8389555294030783, 0.8369968030210323, 0.8282883344611154, 0.8299060161129529, 0.8420666756845904, 0.8401651283159968, 0.8420385921528246], 'MSE_avg': 292839.67669987254, 'MAE_avg': 366.50159584900234, 'R2_avg': 0.8357139152035684, 'MSE_std': 11367.591825656917, 'MAE_std': 5.98414964427825, 'R2_std': 0.006460754403229964}, 'test_perf': {'MSE': 285398.1888637923, 'MAE': 364.22590374727105, 'R2': 0.8406456420906674}, 'params': {'boosting_type': 'dart', 'learning_rate': 0.2, 'n_estimators': 600, 'importance_type': 'split'}}
all = {'training_perf': {'MSE': [257995.30734291868, 235872.42915446268, 228854.43948714103, 227309.92741804483, 242359.6343175312, 250466.07587345774, 243565.04727932138, 223716.3148041926, 225188.51760790712, 232392.88693916934], 'MAE': [337.3668512707934, 328.671475921899, 322.9748424725492, 321.7782408692525, 329.5808356861658, 336.03171911504825, 334.96931933310253, 319.7649115328963, 319.0181378625604, 330.72999837893656], 'R2': [0.8516038860903974, 0.8671626386754436, 0.8696878367626688, 0.8700112733214347, 0.8671802296424392, 0.858711183764292, 0.8666997152727927, 0.8745393777573104, 0.872057854236757, 0.8737971616169571], 'MSE_avg': 236772.05802241465, 'MAE_avg': 328.08863324432036, 'R2_avg': 0.8671451157140494, 'MSE_std': 10918.589919758342, 'MAE_std': 6.508418433245028, 'R2_std': 0.006712682499241215}, 'test_perf': {'MSE': 233079.88811686228, 'MAE': 327.63342055085684, 'R2': 0.8698579831206708}, 'params': {'boosting_type': 'dart', 'learning_rate': 0.2, 'n_estimators': 600, 'importance_type': 'split'}}

out_path = 'dataset/tribunais_trabalho/statistics/distrib_results_categ.png'

df = pd.DataFrame(
                  {
                   'RE': reg['training_perf']['R2'], 
                   'RE + CL': reg_cus['training_perf']['R2'],
                   'RE + CL + PM': all['training_perf']['R2'],
                  }
                 )

sns.set(font_scale=1.3)
ax = sns.boxplot(
                 data=df, 
                 palette="Blues")
ax.set_ylim(0.74, 0.88)
plt.xlabel('Features',labelpad=14)
plt.ylabel('R$^2$-score',labelpad=14)
plt.tight_layout()

sns.scatterplot(x=[0, 1, 2], y=[reg['test_perf']['R2'], 
                                reg_cus['test_perf']['R2'],
                                all['test_perf']['R2']], color='green', s=100)

# plt.show()
plt.savefig(out_path, dpi=400)