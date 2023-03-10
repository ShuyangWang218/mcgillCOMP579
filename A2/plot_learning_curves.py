import matplotlib.pyplot as plt

def plot_learning_curve(average1,average2, average3,std_error1, std_error2, std_error3, e, numberEpisodes, aname):
    x = list(range(0, numberEpisodes))
    plt.plot(average1, color='blue', linewidth=1, label="alpha = 1/4")
    plt.plot(average2, color='red', linewidth=1, label="alpha = 1/8")
    plt.plot(average3, color='green', linewidth=1, label="alpha = 1/16")
    plt.bar(x, std_error1, color='blue')
    plt.bar(x, std_error2, color='red')
    plt.bar(x, std_error3, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average and Standard Error')
    plt.legend(bbox_to_anchor=(0.2, 1), loc='upper center')
    title = aname + ":epsilon = " + str(e)
    plt.title(title)
    fname = title + ".png"
    plt.savefig(fname)
    plt.show()
