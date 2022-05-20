import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TITLE_SIZE = 20

def read_log_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()[1:]

    data = []
    for line in lines:
        if "VIRT" in line:
            break
        d = list(filter(None, line.strip("\n").split(" ")))
        d = d[d.index("spacecl+")+3:d.index("spacecl+")+10]
        
        data.append(d)
    df = pd.DataFrame(data=data, columns=["VIRT", "RES", "SHR", "d1", "CPU%", "MEM", "% Run"]).drop(["d1", "MEM"], axis=1)
    df["VIRT"] = pd.to_numeric(df["VIRT"])
    df["VIRT"] = df['VIRT'].div(1000000)
    for i in range(len(df["RES"])):
        t_formatted = df.loc[i, "RES"]
        if "g" in t_formatted:
            t_formatted = t_formatted[:-1]
        else:
            t_formatted = int(t_formatted)/1000000
        df.loc[i, "RES"] = t_formatted

    df["RES"] = pd.to_numeric(df["RES"])
    df["SHR"] = pd.to_numeric(df["SHR"])
    df["SHR"] = df['SHR'].div(1000)
    df["CPU%"] = df["CPU%"].str.replace(",", ".")
    df["CPU%"] = pd.to_numeric(df["CPU%"])
    for i in range(len(df["% Run"])):
        t_formatted = df.loc[i, "% Run"].split(":")
        t_formatted = int(t_formatted[0])*60 + float(t_formatted[1])
        df.loc[i, "% Run"] = t_formatted
    df["% Run"] = df["% Run"].div(df["% Run"].max()).mul(100)
    df["% Run"] = pd.to_numeric(df["% Run"])

    return df

def get_data(dataframe, verbose=False, maxes_axis=None):
    if len(dataframe[(dataframe['Status']=="ERROR")]) > 0:
        print("There are models including errors:")
        display(dataframe[(dataframe['Status']=="ERROR")])
    
    if verbose:
        print("Including lines:")
        display(dataframe[(dataframe['Status']=="SUCCESS")])
        display(dataframe[(dataframe['Status']=="OUT OF MEMORY")])
        display(dataframe[(dataframe['Status']=="KILLED")])
    
    maxes = []
    maxes_axis_new = []
    for i,file_name in enumerate(dataframe['File']):
        path = f"/home/sara/Documents/Master-thesis/TFLite/memory/{file_name}"
        df_log = read_log_file(path)
        maxes.append(df_log.max())
        if maxes_axis is not None:
            maxes_axis_new.append(maxes_axis[i])

        df_log['File'] = file_name
        if i == 0:
            df_all = df_log
        else:
            df_all = pd.concat([df_all, df_log])

    return df_all, maxes, maxes_axis_new


def analyze(dataframe, axes, plots, legend, max_value, legend_location):
    
    for i, ax in enumerate(axes):
        dataframe.plot(x="% Run", y=plots[i], kind = 'line', ax=ax)	
        max_value[i] = max(max_value[i], max(dataframe[plots[i]]))
        if legend_location is not None:
            ax.legend(legend, loc=legend_location)
        else:
            ax.legend(legend)
    return max_value

def init_plot_memory(plots, x_label, title):
    fig, axes = plt.subplots(1, 4)
    fig.set_figheight(4.5)
    fig.set_figwidth(15)
    fig.suptitle(title, size=TITLE_SIZE)

    for i, ax in enumerate(axes):
        ax.set_xlabel(x_label)
        if plots[i] == 'SHR':
            ax.set_ylabel("MB")
        elif plots[i] == 'CPU%':
            ax.set_ylabel("%")
        else:
            ax.set_ylabel("GB")
        ax.set_title(plots[i])
        ax.yaxis.set_label_coords(-0.12, .52)
    return fig, axes

def plot_analyse(dataframe, legend, title, from_time= 0, to_time=-1, verbose=False, legend_location=None, plot=True):
    plots = ['VIRT', 'RES', 'SHR', 'CPU%'] 
    fig, axes = init_plot_memory(plots, x_label="t", title=title)

    labels = dataframe["File"].unique()
    max_value = [0,0,0,0]
    for i,file_name in enumerate(labels):
        df_log = dataframe[(dataframe['File']==file_name)]
        df_log = df_log[df_log["% Run"]> from_time]
        if to_time != -1:
            df_log = df_log[df_log["% Run"]< to_time]
        max_value = analyze(df_log, axes, plots, legend, max_value, legend_location)

    for i, ax in enumerate(axes):
        ax.set(ylim=(0,max_value[i]+0.2*float(max_value[i])))
    if plot:
        plt.show()
    return axes



def plot_maxes(maxeses, legends, x_axis, x_label, title, plot_diff = False):
    plots = ['VIRT', 'RES', 'SHR', 'CPU%'] 
    fig, axes = init_plot_memory(plots, x_label=x_label, title=title)
    
    all_data = []
    min_per_plot = [1000,10000,10000,10000]
    max_per_plot = [0,0,0,0]
    for i_max, maxes in enumerate(maxeses):
        data = [[], [], [], []]

        for i, line in enumerate(maxes):
            for j,d in enumerate(data):
                d.append(line.values[j])
        
        for i, ax in enumerate(axes):
            if type(x_axis[0]) == list:
                ax.plot(x_axis[i_max], data[i])
            else:
                ax.plot(x_axis, data[i])
            min_per_plot[i] = min(min_per_plot[i], min(data[i]))
            max_per_plot[i] = max(max_per_plot[i], max(data[i]))
        all_data.append(data)

    if plot_diff:
        for i, ax in enumerate(axes):
            diff = np.abs(np.array(all_data[1][i]) - np.array(all_data[0][i]))
            ax.plot(x_axis, diff)
            min_per_plot[i] = min(min_per_plot[i], min(diff))
            max_per_plot[i] = max(max_per_plot[i], max(diff))
            legends.append("diff")
            print(f"Average diff {plots[i]}: {np.average(diff)}")

    for i, ax in enumerate(axes):
        ax.set(ylim=(min(0,min_per_plot[i]), max_per_plot[i]+0.2*float(max_per_plot[i])))
        ax.legend(legends)
        if len(ax.get_xticks()) >= 8:
            every_nth = 2
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
            
    plt.show()



def plot_multi(data, legends, x_label = "x", y_label = "y", title = "-m", plot_diff = False):

    fig, ax = plt.subplots(1,1)
    fig.set_figheight(5)
    fig.set_figwidth(7)
    fig.suptitle(title, size=TITLE_SIZE)

    min_y = 1000
    max_y = 0
    for d in data:
        d.plot(x=x_label, y=y_label, kind = 'line', ax=ax)	
        max_y = max(max_y, max(d[y_label]))
        min_y = min(min_y, min(d[y_label]))
    
    if plot_diff:
        diff = np.abs(data[1][y_label].values -data[0][y_label].values)
        x = list(data[1][x_label].values)
        ax.plot(x, diff)
        legends.append("diff")
        print(f"Average diff: {np.average(diff)}")

    if "(s)" in y_label:
        ax.set_ylabel("seconds")
    else:
        ax.set_ylabel("minutes")
    ax.legend(legends)
    ax.set(ylim=(min(0,min_y), max_y+0.2*float(max_y)))

    plt.show()