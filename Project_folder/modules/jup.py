# libraries 
import os # to set directories 
import numpy as np # for math 
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting figures
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch # for decorative arrows 
from matplotlib.ticker import NullFormatter, StrMethodFormatter # for hiding axes labels
import seaborn as sns # to make figures more aesthetically pleasing
import statsmodels.api as sm # to run OLS regressions

# set directories - making sure we have figures directory for the outputs
cwd = os.getcwd()
figures_dir = os.path.join(cwd, "figures")
os.makedirs(figures_dir, exist_ok=True)

# set up parameters for the figures 
# define the colour palette - using BDO default palette
palette = ["#e81a3b", "#333333", "#5b6e7f", "#008fd2", "#009966"]
font_size = 16
sns.set_style('white')
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "font.family": "Trebuchet MS",
    'axes.grid': False,
    'axes.edgecolor': palette[1],
    'axes.linewidth': 1.0,
    'text.color':  palette[1],
    'axes.labelcolor':  palette[1],
    'xtick.color':  palette[1],
    'ytick.color':  palette[1],
    'axes.titlecolor':  palette[1],
    'font.size': font_size,
    'axes.titlesize': font_size + 4,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    })
# initiate the figure counter
figure_number = 1
figure_part = 1
figure_dpi = 600

# Define Helper functions 
# helper for saving figure parts sequentially 
def save_fig_part(fig, ax, figure_number, figure_part, update_legend=True, dpi=figure_dpi):
    if update_legend:
        plt.legend()
    figname = f"{figures_dir}/Figure_{figure_number}_part{figure_part}.png"
    fig.savefig(f"{figname}", dpi=figure_dpi, bbox_inches='tight')
# helper for adding time series labels 
def add_cartel_labels(x1=50, x2=150, y = 8.8):
    plt.xticks([])
    plt.text(x1, y, 'Pre-cartel period', ha='center', va='bottom')
    plt.text(x2, y, 'Cartel period', ha='center', va='bottom')
    plt.axvline(x=split, color=palette[1])

# DATA GENERATION 

# set the random number seed for reproducibility
np.random.seed(7777)

# number of price points (e.g. if we have monthly prices for 20 years, ~240 price points)
n = 200 
t = np.arange(n)
split = n // 2 # divide into before and during periods
before = t < split 
during = t >= split 

# generate a cost variable - follows a simple AR(1) 
rho = 0.9
cost = 10*np.ones(n) 
eps_cost = np.random.normal(scale=0.5, size=n) # shocks 
for i in range(1, n):
    cost[i] = rho * cost[n-1] + eps_cost[i]
    
cost_spike = 1.5 # we want there to be a cost spike during the cartel to make things interesting
cost[during] += cost_spike

# generate a demand variable (want it to be seasonal with a trend)
demand = 2*np.sin(2*np.pi*t/40) + 0.05*t + np.random.normal(scale=0.5, size=n)

# generate a supply variable (this one can be a random walk)
supply = np.cumsum(np.random.normal(scale=0.05, size=n))

# 'true' model coefficients 
beta0 = 2.0
beta_cost = 1.0
beta_demand = 0.5
beta_supply = -0.6
sigma_eps = 0.1 

# competitive counterfactual price 
eps = np.random.normal(scale=sigma_eps, size=n)
p_comp = beta0 + beta_cost*cost + beta_demand*demand + beta_supply*supply + eps

# add the 'cartel effect' - e.g. a true overcharge of £3
delta = 3 
p_cartel = p_comp.copy()
p_cartel[during] += delta

# now create a 'control' seller for the DiD 
eps2 = np.random.normal(scale=sigma_eps, size=n)
p_control = 0.1*beta0 + beta_cost*cost + beta_demand*demand + beta_supply*supply + eps2

# combine into a df 
df = pd.DataFrame({
    't': t,
    'before': before.astype(int),
    'during': during.astype(int),
    'cost': cost,
    'demand': demand,
    'supply': supply,
    'price_treated': p_cartel,
    'price_treated_counterfactual': p_comp,
    'price_control': p_control
})

# can export the data to csv etc ...

# SHOWING BASELINE OVERCHARGE 

# calculate the true average overcharge in £ and % 
mean_comp = df.loc[df['during'] == 1, 'price_treated_counterfactual'].mean()
mean_during = df.loc[df['during'] == 1, 'price_treated'].mean()
true_avg_overcharge = mean_during - mean_comp
true_avg_overcharge_pct = (true_avg_overcharge/mean_comp)*100


# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f'Figure {figure_number}: The price of apples.')
plt.ylabel('Price (£)')
# placeholder blank labels 
plt.xticks([])
time_txt = plt.text(100, 8.8, r'time $\rightarrow$', ha='center', va='bottom',)

# Part 1: plot the observed price series 
plt.plot(df['t'], df['price_treated'], label='Observed price (£)', color=palette[0], linewidth=2)
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# Part 2: add the cartel labels 
time_txt.remove()
add_cartel_labels()
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 2: plot the true counterfactual price series
plt.plot(df['t'].loc[df['during']==1], df['price_treated_counterfactual'].loc[df['during']==1], color=palette[1], label='Competitive price (£)', linewidth=2)
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# Part 3: shade in the overcharge region
x_during = df.loc[df['during']==1, 't'].values
y1 = df.loc[df['during']==1, 'price_treated'].values
y0 = df.loc[df['during']==1, 'price_treated_counterfactual'].values
plt.fill_between(x_during, y0, y1, alpha=0.2, label='Overcharge region', color=palette[2])
plt.annotate('',xy=(98,16.5),xytext=(50,17.45),arrowprops=dict(color=palette[4], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(50, 17.7, 'True overcharge: ' r'$\delta$' f' = £{true_avg_overcharge:.2f} ({true_avg_overcharge_pct:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)

# display the figure
plt.tight_layout()
plt.show()

# calculate the before and during mean prices 
mean_before = df.loc[df['during'] == 0, 'price_treated'].mean()
mean_during = df.loc[df['during'] == 1, 'price_treated'].mean()
raw_diff = mean_during - mean_before

# AVERAGE PRICE DIFFS 
# update the figure counters
figure_number += 1
figure_part = 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: 'before' vs. 'during' difference in the price of apples (true overcharge = £{true_avg_overcharge:.2f}).")
plt.ylabel('Price (£)')
add_cartel_labels()

# Part 1: plot the observed price series
plt.plot(df['t'], df['price_treated'], label='Observed price (£)', color=palette[0], linewidth=2)
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 2: Add the average price before the cartel 
plt.axhline(y=mean_before, xmax=0.5, linestyle=':', color=palette[2], label="Mean price", linewidth=5)
plt.text(135, mean_before, f"Average 'before' price = £{mean_before:.2f}", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[2], 0.1),edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 3: Add the average price after the cartel
plt.axhline(y=mean_during, xmin=0.5, linestyle=':', color=palette[2], linewidth=5)
plt.text(64, mean_during, f"Average 'during' price = £{mean_during:.2f}", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[2], 0.1),edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# display the figure
plt.tight_layout()
plt.show()

# CONTROL VARIABLES 
# update the figure counters
figure_number += 1
figure_part = 1

# check other controls 
fig, axes = plt.subplots(3,1, sharex=True)

# plot the demand and supply controls
series = [
('cost', '(a) Costs (e.g. fuel/energy costs).'),
('demand', '(b) Demand (e.g. household income).'),
('supply', '(c) Supply (e.g. weather/yield).'),
]

# add labels/aesthetics 
sns.despine(fig, top=True, right=True)  # remove top/right spines
# set ylims 
axes[0].set_ylim(df['cost'].min()-0.3, df['cost'].max())
axes[1].set_ylim(df['demand'].min()-0.5, df['demand'].max())
axes[2].set_ylim(df['supply'].min()-0.05, df['supply'].max())
axes[2].text(50, -0.65, 'Pre-cartel period', ha='center', va='bottom')
axes[2].text(150, -0.65, 'Cartel period', ha='center', va='bottom')
axes[1].yaxis.set_major_formatter(NullFormatter())
axes[2].yaxis.set_major_formatter(NullFormatter())
fig.suptitle(f"Figure {figure_number}: Other variables that might affect the price of apples.", fontsize=font_size+4)
fig.align_ylabels(axes)

# plot 
for ax, (col, title) in zip(axes, series):
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.1f}"))
    ax.set_xlim(-1, 201)
    ax.set_title(title, loc='left', fontsize=font_size)
    ax.set_ylabel(col.capitalize())
    ax.set_xticks([])
    ax.axvline(x=split, color='black')
    ax.plot(df['t'], df[col], linewidth=2, color=palette[2])
    means = df.groupby('during')[col].mean()
    pre_mean = means.get(0, np.nan)
    post_mean = means.get(1, np.nan)
    ax.axhline(y=pre_mean, xmax=0.5, color=palette[0], linewidth=2, linestyle='--')
    ax.axhline(y=post_mean, xmin=0.5, color=palette[0], linewidth=2, linestyle='--')
    save_fig_part(fig, axes, figure_number, figure_part, update_legend=False)
    figure_part += 1
    
# Show the figure
plt.tight_layout()
plt.show()


#FORECASTING METHOD 
# update the figure counters
figure_number += 1
figure_part = 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The forecasting method.")
plt.ylabel('Price (£)')
add_cartel_labels()

# Part 1: Plot the observed price series
plt.plot(df['t'], df['price_treated'], label=r'Observed price (£) $\rightarrow$ $P_t = F(C_t,D_t,S_t) + \varepsilon_t$', color=palette[0], linewidth=2)
save_fig_part(fig, axes, figure_number, figure_part)
figure_part += 1

# Part 2: add step 1 label  
plt.text(49.5, 17.4, 'Forecast method: Step 1\n\n' r'Estimate $F(C_t,D_t,S_t)$ using the pre-cartel data,' '\n' r'e.g. $F(C_t,D_t,S_t) \approx \alpha + \beta_CC_t + \beta_DD_t + \beta_SS_T.$ ', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
arrow = FancyArrowPatch((49.5,15.45), (49.5,16.1), arrowstyle='-', lw=1, color=palette[3])
ax.add_patch(arrow)
arrow = FancyArrowPatch((5,15.5), (94,15.5), arrowstyle='<->', lw=1, color=palette[3], mutation_scale=15)
ax.add_patch(arrow)
save_fig_part(fig, axes, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 3: add step 2 label
plt.text(149.5, 13.5, 'Forecast method: Step 2\n\n' r'Use $\hat{\alpha}, \hat{\beta_C}, \hat{\beta_D}, \hat{\beta_S}$, and the during-cartel data' '\n' r'to predict prices, $\hat{P_t}$, in the cartel period.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
arrow = FancyArrowPatch((149.5,15), (149.5,15.55), arrowstyle='-', lw=1, color=palette[3])
ax.add_patch(arrow)
arrow = FancyArrowPatch((106,15.5), (195,15.5), arrowstyle='<->', lw=1, color=palette[3], mutation_scale=15)
ax.add_patch(arrow)
save_fig_part(fig, axes, figure_number, figure_part, update_legend=False)

# show the figure
plt.tight_layout()
plt.show()

# define the data matrices - before period
X_before = sm.add_constant(df.loc[df['during']==0, ['cost', 'demand', 'supply']])
y_before = df.loc[df['during']==0, 'price_treated']
# estimate the regression model on the before period 
model_before = sm.OLS(y_before, X_before).fit()

# define the data matrix (all periods)
X_all = sm.add_constant(df[['cost', 'demand', 'supply']])
# use the estimated model to predict the price series in the cartel period 
df['forecast_price'] = model_before.predict(X_all)

# Use the predicted series to calculate the predicted overcharge in £ and % 
mean_fcst = df.loc[df['during'] == 1, 'forecast_price'].mean()
fcst_overcharge = mean_during - mean_fcst
fcst_overcharge_pct = (fcst_overcharge/mean_fcst)*100

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The forecasting method.")
plt.ylabel('Price (£)')
add_cartel_labels()

# Part 1: Plot the observed price series
plt.plot(df['t'], df['price_treated'], label=r'Observed price (£) $\rightarrow$ $P_t = F(C_t,D_t,S_t) + \varepsilon_t$', color=palette[0], linewidth=2)
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'forecast_price'], label=r'Predicted price (£) $\rightarrow$ $\hat{P_t}$', color=palette[2], linewidth=2)
plt.annotate('',xy=(150,13.9),xytext=(150,12.2),arrowprops=dict(color=palette[2], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(150, 11.3, 'Predicted overcharge: ' r'$\hat{\delta}$' f' = £{fcst_overcharge:.2f} ({fcst_overcharge_pct:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[2], 0.1),edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# Part 2: Show the true overcharge 
plt.fill_between(x_during, y0, y1, alpha=0.2, label='True overcharge region', color=palette[1])
plt.annotate('',xy=(98,16.5),xytext=(50,17.45),arrowprops=dict(color=palette[4], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(50, 17.7, 'True overcharge: ' r'$\delta$' f' = £{true_avg_overcharge:.2f} ({true_avg_overcharge_pct:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)

# show the figure
plt.tight_layout()
plt.show()

# DUMMY VARIABLE METHOD 

# Generate the cartel dummy 
df['cartel_dummy'] = df['during']
# Create the data matrices for the regression - this time using the full sample
X_full = sm.add_constant(df[['cartel_dummy','cost', 'demand', 'supply']])
y_full = df['price_treated']
# estimate the full regression model
model_full = sm.OLS(y_full, X_full).fit()

# Create the counterfactual price series by subtracting the cartel dummy from the observed prices during the cartel period
df['price_treated_dummy'] = df['price_treated']
df.loc[df['during']==1, 'price_treated_dummy'] -= model_full.params[1]

# calculate the overcharge 
mean_dummy = df.loc[df['during'] == 1, 'price_treated_dummy'].mean()
dmmy_overcharge = mean_during - mean_dummy
dmmy_overcharge_pct = (dmmy_overcharge/mean_dummy)*100

# update the figure counters
figure_number += 1
figure_part = 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The dummy variable method.")
plt.ylabel('Price (£)')
add_cartel_labels()

# Part 1: Plot the observed price series
plt.plot(df['t'], df['price_treated'], label=r'Observed price (£) $\rightarrow$ $P_t = F(C_t,D_t,S_t) + \varepsilon_t$', linewidth=2, color=palette[0])
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# Step 2: add description of the method 
dum_text = plt.text(49.5, 17.4, 'Dummy variable method\n\n' r'Estimate $F(C_t,D_t,S_t)$ using the full dataset,' '\n' r'e.g. $F(C_t,D_t,S_t) \approx \alpha + \beta_CC_t + \beta_DD_t + \beta_SS_T.$ ', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
arrow1 = FancyArrowPatch((49.5,15.45), (49.5,16.1), arrowstyle='-', lw=1, color=palette[3])
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((5,15.5), (194,15.5), arrowstyle='<->', lw=1, color=palette[3], mutation_scale=15)
ax.add_patch(arrow2)
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# step 3: add warning text 
but_text = plt.text(149.5, 13, r'But ... $F(\cdot) \approx \alpha + \beta_CC_t + \beta_DD_t + \beta_SS_T.$' '\n\ndoes not capture the cartel effect.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# step 4: add solution text 
but_text.remove()
solu_text = plt.text(149.5, 13, 'Solution\n\n' r'Include a dummy variable, $Cartel_t$, so that' '\n\n' r'$F(\cdot) \approx \alpha + \delta Cartel_t + \beta_CC_t + \beta_DD_t + \beta_SS_T.$', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# step 5: add how it works text and line shift 
solu_text.remove()
dum_text.remove()
dum_text = plt.text(49.5, 17.4, 'Dummy variable method\n\n' r'Estimate $F(C_t,D_t,S_t)$ using the full dataset,' '\n' r'$F(\cdot) \approx \alpha + \delta Cartel_t + \beta_CC_t + \beta_DD_t + \beta_SS_T.$ ', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
how_text = plt.text(149.5, 13, 'How it works\n\n' r'$\delta$ is the average change in price' '\n' r'that cannot be explained by: $C_t, D_t, S_t$.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Step 6: show a shift due to delta
arrow1.remove()
arrow2.remove()
how_text.remove()
how_text = plt.text(149.5, 11.7, 'How it works\n\n' r'We subtract $\hat{\delta}$ from the observed price' '\n' r'to give a predicted price: $\hat{P_t} = P_t - \hat{\delta}$.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated']-3.1, label=r'Predicted price (£) $\rightarrow$ $\hat{P_t} = P_t - \hat{\delta}$', linewidth=2, color=palette[2])
arrow3 = FancyArrowPatch((130,17), (130,19), arrowstyle='<->', lw=1, color=palette[1], mutation_scale=15)
ax.add_patch(arrow3)
plt.text(133, 18, r'$\hat{\delta}$', ha='center', va='center')
arrow3 = FancyArrowPatch((172,17.6), (172,19.6), arrowstyle='<->', lw=1, color=palette[1], mutation_scale=15)
ax.add_patch(arrow3)
plt.text(175, 18.6, r'$\hat{\delta}$', ha='center', va='center')
save_fig_part(fig, ax, figure_number, figure_part)

# Show the figure
plt.tight_layout()
plt.show()

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The dummy variable method.")
plt.ylabel('Price (£)')
add_cartel_labels()

# Part 1: Plot the observed price series
plt.plot(df['t'], df['price_treated'], label=r'Observed price (£) $\rightarrow$ $P_t = F(C_t,D_t,S_t) + \varepsilon_t$', linewidth=2, color=palette[0])
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated_dummy'], label=r'Predicted price (£) $\rightarrow$ $P_t - \hat{\delta}$', color=palette[2], linewidth=2)
plt.annotate('',xy=(150,13.9),xytext=(150,12.22),arrowprops=dict(color=palette[2], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(150, 11.3, 'Predicted overcharge: ' r'$\hat{\delta}$' f' = £{dmmy_overcharge:.2f} ({dmmy_overcharge_pct:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[2], 0.1),edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# Part 2: Show the true overcharge 
plt.fill_between(x_during, y0, y1, alpha=0.2, label='True overcharge region', color=palette[1])
plt.annotate('',xy=(98,16.5),xytext=(50,17.45),arrowprops=dict(color=palette[4], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(50, 17.7, 'True overcharge: ' r'$\delta$' f' = £{true_avg_overcharge:.2f} ({true_avg_overcharge_pct:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)

# Show the figure
plt.tight_layout()
plt.show()

# EXAMPLE WHERE DUMMY VARIABLE METHOD DOES NOT WORK 

# now just add a final example to show that if you do not use the correct controls it won't work 
# Create the data matrices for the regression - this time using the full sample
X_full_incorrect = sm.add_constant(df[['cartel_dummy','cost']])
# estimate the full regression model
model_full_incorrect = sm.OLS(y_full, X_full_incorrect).fit()

# Create the counterfactual price series by subtracting the cartel dummy from the observed prices during the cartel period
df['price_treated_dummy_incorrect'] = df['price_treated']
df.loc[df['during']==1, 'price_treated_dummy_incorrect'] -= model_full_incorrect.params[1]

# calculate the overcharge 
mean_dummy_incorrect = df.loc[df['during'] == 1, 'price_treated_dummy_incorrect'].mean()
dmmy_overcharge_incorrect = mean_during - mean_dummy_incorrect
dmmy_overcharge_pct_incorrect = (dmmy_overcharge_incorrect/mean_dummy_incorrect)*100

# update the figure counters
figure_number += 1
figure_part = 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The dummy variable method.")
plt.ylabel('Price (£)')
add_cartel_labels()

# Part 1: Plot the observed price series
plt.plot(df['t'], df['price_treated'], label=r'Observed price (£) $\rightarrow$ $P_t = F(C_t,D_t,S_t) + \varepsilon_t$', linewidth=2, color=palette[0])
dum_text = plt.text(49.5, 17.4, 'Dummy variable method\n\n' r'Estimate $F(C_t,D_t,S_t)$ using the full dataset,' '\n' r'$F(\cdot) \approx \alpha + \delta Cartel_t + \beta_CC_t.$ ', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# part 2: show the predicted price series and overcharge 
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated_dummy_incorrect'], label=r'Predicted price (£) $\rightarrow$ $P_t - \hat{\delta}$', color=palette[2], linewidth=2)
plt.annotate('',xy=(150,12),xytext=(150,11.25),arrowprops=dict(color=palette[2], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(150, 10.3, 'Predicted overcharge: ' r'$\hat{\delta}$' f' = £{dmmy_overcharge_incorrect:.2f} ({dmmy_overcharge_pct_incorrect:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[2], 0.1),edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# Part 2: Show the true overcharge 
plt.fill_between(x_during, y0, y1, alpha=0.2, label='True overcharge region', color=palette[1])
plt.annotate('',xy=(98,16.5),xytext=(50,17.45),arrowprops=dict(color=palette[4], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(50, 17.7, 'True overcharge: ' r'$\delta$' f' = £{true_avg_overcharge:.2f} ({true_avg_overcharge_pct:.1f}%).', ha='center', va='bottom', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
dum_text.remove()
save_fig_part(fig, ax, figure_number, figure_part)

# Show the figure
plt.tight_layout()
plt.show()

# DIFFERENCE IN DIFFERENCES 

# Calculate the placebo DiD to confirm that the pre-trends are parallel 
# construct the panel dataset 
pre_panel = pd.concat([
    pd.DataFrame({"t": df.loc[df['during']==0, "t"], "price": df.loc[df['during']==0, "price_treated"], "treated": 1}),
    pd.DataFrame({"t": df.loc[df['during']==0, "t"], "price": df.loc[df['during']==0, "price_control"], "treated": 0}) 
], ignore_index=True)
# add the interaction term
pre_panel['post'] = np.where(pre_panel['t'] > 50, 1, 0)
pre_panel["interaction"] = pre_panel["post"] * pre_panel["treated"]

# create the data matrices for the regression 
X_pl_did = sm.add_constant(pre_panel[["post", "treated", "interaction"]])
y_pl_did = pre_panel["price"]

# estimate the basic model without controls
did_pl = sm.OLS(y_pl_did, X_pl_did).fit()
did_est_pl = did_pl.params["interaction"]

# print the results (should be close to 0)
print(f"The estimated difference in gradients is: {did_est_pl:.3f}")

# DIFF IN DIFF PRE TRENDS  
# update the figure counters
figure_number += 1
figure_part = 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
add_cartel_labels(y=6.9)

# Part 1: treated price series 
plt.plot(df['t'], df['price_treated'], label='Observed price (£) - Cartelists', linewidth=2, color=palette[0])
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# part 2: control price series 
plt.plot(df['t'], df['price_control'], label='Observed price (£) - Non-cartelists', linewidth=2, color=palette[2])
save_fig_part(fig, ax, figure_number, figure_part)

# Show the figure
plt.tight_layout()
plt.show()

# DIFF IN DIFF AVERAGE PRICE DIFFERENCES 

# Step 2: now illustrate visually, showing parallell trends first 

# calculate the trend to check whether it is parallel
sl1, int1 = np.polyfit(df.loc[df['during']==0, 't'], df.loc[df['during']==0, 'price_treated'], 1)
sl2, int2 = np.polyfit(df.loc[df['during']==0, 't'], df.loc[df['during']==0, 'price_control'], 1)
sl3, int3 = np.polyfit(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated'], 1)
sl4, int4 = np.polyfit(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_control'], 1)

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=6.9)

# Part 3: treated & control price series 
plt.plot(df.loc[df['during']==0,'t'], df.loc[df['during']==0,'price_treated'], label='Observed price (£) - Cartelists', linewidth=2, color=palette[0], alpha=0.3)
plt.plot(df.loc[df['during']==0,'t'], df.loc[df['during']==0,'price_control'], label='Observed price (£) - Non-cartelists', linewidth=2, color=palette[2], alpha=0.3)
p1, = plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated'], linewidth=2, color=palette[0], alpha=0.3)
p2, = plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_control'], linewidth=2, color=palette[2], alpha=0.3)
plt.plot(df.loc[df['during']==0, 't'], int1 + sl1*df.loc[df['during']==0, 't'], linewidth=2, color=palette[0], label="Price trend - Cartelists")
ax.legend(loc = 'upper left')
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 4: add the trend lines - before, control
plt.plot(df.loc[df['during']==0, 't'], int2 + sl2*df.loc[df['during']==0, 't'], linewidth=2, color=palette[2], label="Price trend - Non-cartelists")
plt.text(50, 16, 'Both sets of prices evolve\nin parallel before the cartel.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
ax.legend(loc = 'upper left')
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 5: add text
plt.text(150, 10, "We assume that, but-for the cartel,\nthe cartelist's prices would follow\nthe same trend as the non-cartelists.", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 6: add differences 
p1.remove()
p2.remove()
arrow1 = FancyArrowPatch((150,11.2), (150,14), arrowstyle='->', lw=2, color=palette[1], mutation_scale=15)
ax.add_patch(arrow1)
plt.text(150, 15 , "This implies that\n" r"$\Delta$ Cartelists' price = $\Delta$ Non-cartelists' price", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()

# Step 3: now remove the trend lines and show the mean differences instead 

# calculate the four means
treated_before_mean = df.loc[df['during']==0, 'price_treated'].mean()
treated_during_mean = df.loc[df['during']==1, 'price_treated'].mean()
control_before_mean = df.loc[df['during']==0, 'price_control'].mean()
control_during_mean = df.loc[df['during']==1, 'price_control'].mean()

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=6.9)

# Part 7: control price series and averages
plt.plot(df.loc[df['during']==0,'t'], df.loc[df['during']==0,'price_treated'], label='Observed price (£) - Cartelists', linewidth=2, color=palette[0], alpha=0.3)
plt.plot(df.loc[df['during']==0,'t'], df.loc[df['during']==0,'price_control'], label='Observed price (£) - Non-cartelists', linewidth=2, color=palette[2], alpha=0.3)
plt.axhline(y=treated_before_mean, xmax=0.5, color=palette[0], label="Average price (£) - Cartelists", linewidth=2)
plt.axhline(y=control_before_mean, xmax=0.5, color=palette[2], label="Average price (£) - Non-cartelists", linewidth=2)
plt.text(50, 16, 'Step 1: calculate the average\nprice before the cartel.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
ax.legend(loc = 'upper left')
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 8: treated price series and averages 
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated'], linewidth=2, color=palette[0], alpha=0.3)
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_control'], linewidth=2, color=palette[2], alpha=0.3)
plt.axhline(y=treated_during_mean, xmin=0.5, color=palette[0], linewidth=2)
plt.axhline(y=control_during_mean, xmin=0.5, color=palette[2], linewidth=2)
plt.text(150, 11, 'Step 2: and during the cartel.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()

# DIFF IN DIFF VISUALISING THE EFFECTS STEP BY STEP 

# stepwise DiD estimator construction 

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=6.9)

# Part 9: treated & control price series 
plt.axhline(y=treated_before_mean, xmax=0.5, color=palette[0], label="Average price (£) - Cartelists", linewidth=2)
plt.axhline(y=control_before_mean, xmax=0.5, color=palette[2], label="Average price (£) - Non-cartelists", linewidth=2)
plt.axhline(y=treated_during_mean, xmin=0.5, color=palette[0], linewidth=2)
plt.axhline(y=control_during_mean, xmin=0.5, color=palette[2], linewidth=2)
ax.legend(loc = 'upper left')
txt = plt.text(40, 18, r'How do the average prices compare?', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 10: add a label for the pre-cartel mean for the non-cartelists 
pre_nc_txt = plt.text(50, control_before_mean, r"Non-cartelists 'before' = $\alpha$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 11: add label for the pre-cartel men for the cartelists
pre_c_txt = plt.text(50, treated_before_mean, r"Cartelists 'before' = $\alpha + \gamma$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[0],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 12: add a label for the during cartel mean for the non-cartelists 
during_nc_txt = plt.text(150, control_during_mean, r"Non-cartelists 'during' = $\alpha + \lambda$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[2],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 13: add a label for the during cartel mean for the cartelists 
during_c_txt = plt.text(150, treated_during_mean, r"Cartelists 'during' = $\alpha + \gamma + \lambda$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[0],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 14: now adjust the label to include the cartel effect 
during_c_txt.remove()
during_c_txt = plt.text(150, treated_during_mean, r"Cartelists 'during' = $\alpha + \gamma + \lambda + \delta$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[0],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()

# DIFF IN DIFF ESTIMATOR IN NOTATION FORM 
# Step 4: now remove the line itself and overlay the respective areas for the DiD

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=6.9)

# Part 9: treated & control price series 
plt.axhline(y=treated_before_mean, xmax=0.5, color=palette[0], label="Average price (£) - Cartelists", linewidth=2)
plt.axhline(y=control_before_mean, xmax=0.5, color=palette[2], label="Average price (£) - Non-cartelists", linewidth=2)
plt.axhline(y=treated_during_mean, xmin=0.5, color=palette[0], linewidth=2)
plt.axhline(y=control_during_mean, xmin=0.5, color=palette[2], linewidth=2)
pre_nc_txt = plt.text(50, control_before_mean, r"Non-cartelists 'before' = $\alpha$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[2],linewidth=1))
pre_c_txt = plt.text(50, treated_before_mean, r"Cartelists 'before' = $\alpha + \gamma$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[0],linewidth=1))
during_nc_txt = plt.text(150, control_during_mean, r"Non-cartelists 'during' = $\alpha + \lambda$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[2],linewidth=1))
during_c_txt = plt.text(150, treated_during_mean, r"Cartelists 'during' = $\alpha + \gamma + \lambda + \delta$", ha='center', va='center', bbox=dict(boxstyle="round, pad=0.5",facecolor="white",edgecolor=palette[0],linewidth=1))
ax.legend(loc = 'upper left')
stp3_txt = plt.text(40, 18, r'Step 3: Calculate the $\Delta$ in prices.', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# Part 10: add more text labels 
arrow1 = FancyArrowPatch((95,treated_before_mean+0.15), (95,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[0], mutation_scale=15)
ax.add_patch(arrow1)
plt.text(91.5, (treated_before_mean+treated_during_mean)/2 , r"$\Delta$ Cartelists' price = $\gamma + \delta$", ha='right', va='center', color=palette[0])
arrow2 = FancyArrowPatch((105,control_before_mean+0.15), (105,control_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[2], mutation_scale=15)
ax.add_patch(arrow2)
plt.text(108.5, (control_before_mean+control_during_mean)/2 , r"$\Delta$ Non-cartelists' price = $\gamma$", ha='left', va='center', color = palette[2])
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 11: add step 4: calc DiD
stp3_txt.remove()
notice_txt = plt.text(150, 10, "But ... we assumed that\n" r"$\Delta$ Cartelists' price = $\Delta$ Non-cartelists' price", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 12: add final explanatory text 
notice_txt.remove()
notice_txt = plt.text(150, 10, "This means the 'Difference-in-Differences'\n" r"$\Delta$ Cartelists' price - $\Delta$ Non-cartelists' price, " '\n' r"$\delta$, is equal to the cartel overcharge.", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 13: add step 4 text
notice_txt.remove()
plt.text(150, 10, "Step 4: Calculate the\n 'Difference-in-Differences'\n" r"$(\gamma + \delta) - \gamma$", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()

# Step 5, now move the Non-cartelists' price diff upwards to show that the difference between the two is the DiD estimate. 

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=6.9)

# Part 12: treated & control price series 
plt.axhline(y=treated_before_mean, xmax=0.5, color=palette[0], label="Average price (£) - Cartelists", linewidth=2)
plt.axhline(y=treated_during_mean, xmin=0.5, color=palette[0], linewidth=2)
ax.legend(loc = 'upper left')
arrow1 = FancyArrowPatch((95,treated_before_mean+0.15), (95,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[0], mutation_scale=15)
ax.add_patch(arrow1)
plt.text(91.5, (treated_before_mean+treated_during_mean)/2 , r"$\Delta$ Cartelists' price = $\gamma + \delta$", ha='right', va='center', color=palette[0])
plt.text(150, 10, "Step 4: Calculate the\n 'Difference-in-Differences'\n" r"$(\gamma + \delta) - \gamma$", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
arrow2 = FancyArrowPatch((105,treated_before_mean+0.15), (105,treated_before_mean+(control_during_mean-control_before_mean)-0.15), arrowstyle='<->', lw=2, color=palette[2], mutation_scale=15)
ax.add_patch(arrow2)
plt.text(108.5, (2*treated_before_mean+(control_during_mean-control_before_mean))/2 , r"$\Delta$ Non-cartelists' price = $\gamma$", ha='left', va='center', color=palette[2])
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 13: add the DiD estimate
arrow3 = FancyArrowPatch((105,treated_before_mean+(control_during_mean-control_before_mean)), (105,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[1], mutation_scale=15)
ax.add_patch(arrow3)
plt.text(108.5, ((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2 , r"Difference-in-Differences = $\delta$", ha='left', va='center', color=palette[1])
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()

# CALC THE DIFF IN DIFF ESTIMATE 
# Step 6: Show the estimated overcharge amounts 
# calculate the raw DiD estimate 
did_est = (treated_during_mean - treated_before_mean) - (control_during_mean - control_before_mean)
print(f"The raw DiD estimate of the price overcharge is: £{did_est:.2f}")

# OVERLAY DIFF IN DIFF ESTIMATE ON THE FIGURE 
# Step 7: add the differences and the overcharge estimate to the figure 

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=6.9)

# Part 14: treated & control price series - add the numeric values instead of the text labels 
plt.axhline(y=treated_before_mean, xmax=0.5, color=palette[0], label="Average price (£) - Cartelists", linewidth=2)
plt.axhline(y=treated_during_mean, xmin=0.5, color=palette[0], linewidth=2)
ax.legend(loc = 'upper left')
arrow1 = FancyArrowPatch((95,treated_before_mean+0.15), (95,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[0], mutation_scale=15)
ax.add_patch(arrow1)
plt.text(91.5, (treated_before_mean+treated_during_mean)/2 , f"£{treated_during_mean-treated_before_mean:.2f}", ha='right', va='center', color=palette[0])
arrow2 = FancyArrowPatch((105,treated_before_mean+0.15), (105,treated_before_mean+(control_during_mean-control_before_mean)-0.15), arrowstyle='<->', lw=2, color=palette[2], mutation_scale=15)
ax.add_patch(arrow2)
plt.text(108.5, (2*treated_before_mean+(control_during_mean-control_before_mean))/2 , f"£{control_during_mean-control_before_mean:.2f}", ha='left', va='center', color=palette[2])
arrow3 = FancyArrowPatch((105,treated_before_mean+(control_during_mean-control_before_mean)), (105,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[1], mutation_scale=15)
ax.add_patch(arrow3)
plt.text(108.5, ((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2 , f"£{did_est:.2f}", ha='left', va='center', color=palette[1])
plt.text(150, 10, "Step 4: Calculate the\n 'Difference-in-Differences'\n" r"$(\gamma + \delta) - \gamma$", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)
figure_part += 1

# part 15: add overcharge annotation 
plt.annotate('',xy=(121,((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2),xytext=(134,((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2),arrowprops=dict(color=palette[4], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(150, ((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2, r'$\hat{\delta}$ = Overcharge', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()

# PARALLEL TRENDS VIOLATION 
# Create a new price series for the control firms that violates parallel trends
# adjust cost, demand, and supply 
df['cont_cost'] = cost + 0.6*np.linspace(0, 3.0, n)
df['cont_demand'] = demand + 0.3*np.sin(2*np.pi*t/30.0)
df['cont_supply'] = supply + np.cumsum(np.random.normal(0, 0.01, n))
df["price_control_new"] = -0.5*beta0 + beta_cost*df['cont_cost'] + beta_demand*df['cont_demand'] + beta_supply*df['cont_supply'] + eps2

# Then calculate the placebo DiD to confirm that the pre-trends are no longer parallel 
# construct the panel dataset 
pre_panel = pd.concat([
    pd.DataFrame({"t": df.loc[df['during']==0, "t"], "price": df.loc[df['during']==0, "price_treated"], "treated": 1}),
    pd.DataFrame({"t": df.loc[df['during']==0, "t"], "price": df.loc[df['during']==0, "price_control_new"], "treated": 0}) 
], ignore_index=True)
# add the interaction term
pre_panel['post'] = np.where(pre_panel['t'] > 50, 1, 0)
pre_panel["interaction"] = pre_panel["post"] * pre_panel["treated"]
# create the data matrices for the regression 
X_pl_did = sm.add_constant(pre_panel[["post", "treated", "interaction"]])
y_pl_did = pre_panel["price"]
# estimate the basic model without controls
did_pl = sm.OLS(y_pl_did, X_pl_did).fit()
did_est_pl = did_pl.params["interaction"]

# print the results (should be non-zero)
print(f"The estimated difference in gradients is: {did_est_pl:.3f}")

#  VISUALISE IT 
# Step 1: Now show what happens when parallel trends is violated 
# update the figure counters
figure_number += 1 
figure_part = 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
ax.set_ylim(df['price_control_new'].min()-0.5,df['price_treated'].max()+0.5)
add_cartel_labels(y=5.9)

# Part 1: treated price series 
plt.plot(df['t'], df['price_treated'], label='Observed price (£) - Cartelists', linewidth=2, color=palette[0])
plt.plot(df['t'], df['price_control_new'], label='Observed price (£) - Non-cartelists', linewidth=2, color=palette[2])
save_fig_part(fig, ax, figure_number, figure_part)

# Show the figure
plt.tight_layout()
plt.show()

# SHOW THE TREND LINES ON IT 
# Step 2: Now show what happens when parallel trends is violated 

# calculate the trend to check whether it is parallel
sl1, int1 = np.polyfit(df.loc[df['during']==0, 't'], df.loc[df['during']==0, 'price_treated'], 1)
sl2, int2 = np.polyfit(df.loc[df['during']==0, 't'], df.loc[df['during']==0, 'price_control_new'], 1)

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
ax.set_ylim(df['price_control_new'].min()-0.5,df['price_treated'].max()+0.5)
add_cartel_labels(y=5.9)

# Part 2: treated price series + pre + post trends
plt.plot(df.loc[df['during']==0,'t'], df.loc[df['during']==0,'price_treated'], label='Observed price (£) - Cartelists', linewidth=2, color=palette[0], alpha=0.3)
plt.plot(df.loc[df['during']==0,'t'], df.loc[df['during']==0,'price_control_new'], label='Observed price (£) - Non-cartelists', linewidth=2, color=palette[2], alpha=0.3)
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_treated'], linewidth=2, color=palette[0])
plt.plot(df.loc[df['during']==1, 't'], df.loc[df['during']==1, 'price_control_new'], linewidth=2, color=palette[2])
plt.plot(df.loc[df['during']==0, 't'], int1 + sl1*df.loc[df['during']==0, 't'], linewidth=2, color=palette[0], label="Price trend - Cartelists")
plt.plot(df.loc[df['during']==0, 't'], int2 + sl2*df.loc[df['during']==0, 't'], linewidth=2, color=palette[2], label="Price trend - Non-cartelists")
plt.text(50, 16, "In this example, the parallel\ntrend assumption is violated.", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)
figure_part += 1

# part 3: add final text
plt.text(150, 10, "Consequently, the DiD\n" r"estimate, $\hat{\delta}$, will be biased.", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part)

# Show the figure
plt.tight_layout()
plt.show()

# CALUCLATE THE NEW DIFF IN DIFF ESTIMATE 
# plot 

# update the figure counters
figure_part += 1

# Create the figure
fig, ax = plt.subplots()
sns.despine(ax=ax, top=True, right=True)  # remove top/right spines
ax.set_xlim(-1,201)
ax.set_ylim(df['price_control_new'].min()-0.5,df['price_treated'].max()+0.5)
plt.title(f"Figure {figure_number}: The Difference-in-Difference (DiD) method.")
plt.ylabel('Price (£)')
add_cartel_labels(y=5.9)

# Part 13: treated & control price series - add the numeric values instead of the text labels 
plt.axhline(y=treated_before_mean, xmax=0.5, color=palette[0], label="Average price (£) - Cartelists", linewidth=2)
plt.axhline(y=treated_during_mean, xmin=0.5, color=palette[0], linewidth=2)
ax.legend(loc = 'upper left')
arrow1 = FancyArrowPatch((95,treated_before_mean+0.15), (95,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[0], mutation_scale=15)
ax.add_patch(arrow1)
plt.text(91.5, (treated_before_mean+treated_during_mean)/2 , f"£{treated_during_mean-treated_before_mean:.2f}", ha='right', va='center', color=palette[0])
arrow2 = FancyArrowPatch((105,treated_before_mean+0.15), (105,treated_before_mean+(control_during_mean-control_before_mean)-0.15), arrowstyle='<->', lw=2, color=palette[2], mutation_scale=15)
ax.add_patch(arrow2)
plt.text(108.5, (2*treated_before_mean+(control_during_mean-control_before_mean))/2 , f"£{control_during_mean-control_before_mean:.2f}", ha='left', va='center', color=palette[2])
arrow3 = FancyArrowPatch((105,treated_before_mean+(control_during_mean-control_before_mean)), (105,treated_during_mean-0.15), arrowstyle='<->', lw=2, color=palette[1], mutation_scale=15)
ax.add_patch(arrow3)
plt.text(108.5, ((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2 , f"£{did_est:.2f}", ha='left', va='center', color=palette[1])
plt.text(150, 10, "In this example, using DiD\nunderestimates the overcharge.", ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[3], 0.1),edgecolor=palette[3],linewidth=1))
plt.annotate('',xy=(121,((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2),xytext=(134,((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2),arrowprops=dict(color=palette[4], shrink=0.02, width=0.5, headwidth=8), ha='center', va='center')
plt.text(150, ((treated_before_mean+(control_during_mean-control_before_mean))+(treated_during_mean-0.15))/2, r'$\hat{\delta}$ = Overcharge', ha='center', va='center', bbox=dict(boxstyle="square, pad=0.5",facecolor=mcolors.to_rgba(palette[4], 0.1),edgecolor=palette[4],linewidth=1))
save_fig_part(fig, ax, figure_number, figure_part, update_legend=False)

# Show the figure
plt.tight_layout()
plt.show()



