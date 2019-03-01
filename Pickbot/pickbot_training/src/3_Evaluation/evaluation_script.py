import matplotlib.pyplot as plt
import numpy as np
import csv

Episode_Steps_initial=[]
Episode_Rewards_initial = []
Episode_initial = []


Episode_Steps=[]
Episode_Rewards = []
Episode = []
Reached_Goal=[]






def reshape_lists(steps):
    summe=0
    for i in range(len(Episode_Steps_initial)):
        summe +=Episode_Steps_initial[i]
        if summe >=steps:
            summe-=Episode_Steps_initial[i]
            break
        else:
            Episode_Steps.append(Episode_Steps_initial[i])
            Episode_Rewards.append(Episode_Rewards_initial[i])
            Episode.append(Episode_initial[i])
    """
    summe=0
    for i in range(len(Episode_Steps)):
        summe +=Episode_Steps[i]

    print ("Summe der Steps nach reshape: "+str(summe))
    print ("Summe episoden: "+str(len(Episode_Steps)))
    """


def read(filename):
    with open(filename, 'r') as file:
        for line in file:
            data=line.strip().split(";")
            Episode_Rewards_initial.append(float(data[0]))
            Episode_initial.append(int(data[1]))
            Episode_Steps_initial.append(int(data[2]))


def reached_goal():
    #create list reached goal with 1 for goal reached and 0 for goal not reached
    #goal is considered to be reached when reward is over 500 and if steps are below 120 which is the maximum steps amount each episode
    

    for i in range(len(Episode_Rewards)):
        if Episode_Rewards[i]>500 and Episode_Steps[i]<120 :
            Reached_Goal.append(1)
        else:
            Reached_Goal.append(0)
    return Reached_Goal

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def create_range(x):
    #create list with correkt episodes for the average values to plot. Starting at Episode 100 as the first one
    episode_range=[]
    for i in range(x+1):
        if i>=100:
            episode_range.append(i)
    return episode_range

if __name__ == '__main__':
    #Define a Name to describe the Graphs
    Trainingname="Baseline-DeepQ_Model-learned-with-Random-Spawn-applied-to-without-Random-Spawn_100000"
    #Define Name of the CSV File with Trainingsresults in 3_Evaluation Folder
    filename ="result_logger_2019-03-01 10:46:22.355174.csv"
    #Define Average over a number of Episodes
    average_over=100
    #Trainingssteps
    steps=100000



    #read CSV FILE  IMPORTANT THE NAME NEEDS TO BE CHANGED IN THE READ FUNCTION TO MATCH THE RIGHT CSV FILE IN THE EVALUATION FOLDER
    read(filename)
    #Reshape lists to a 100000 Steps 
    reshape_lists(steps)
    #Creates a list wich defines if an episode reached its goal
    reached_goal()
    
    #Plot Episode Reward 
    plt.plot(Episode,Episode_Rewards)
    plt.xlabel('Episoden')
    plt.ylabel('Reward/Episode')
    plt.grid(True)
    plt.xlim((0, len(Episode)))
    plt.ylim((-300, 1400))
    plt.savefig(Trainingname+'_Episode_reward.png')
    plt.close()
    
    #Plot Average Episode Reward
    x=create_range(len(Episode_Rewards))
    y_Reward= running_mean(Episode_Rewards,average_over)
    plt.plot(x,y_Reward)
    plt.xlabel('Episoden')
    plt.ylabel('Durchschnittlicher Reward/Episode (100 Episoden)')
    plt.grid(True)
    plt.xlim((average_over, len(Episode)))
    plt.ylim((-300, 550))
    plt.savefig(Trainingname+'_Average_Episode_reward.png')
    plt.close()
    
    #Plot Success Rate
    plt.plot(Episode,Reached_Goal)
    plt.xlabel('Episoden')
    plt.ylabel('Erfolgsrate')
    plt.grid(True)
    plt.xlim((0, len(Episode)))
    plt.ylim((0, 1))
    plt.savefig(Trainingname+'_Success_Rate.png')
    plt.close()

    #Plot Average Success Rate 
    x=create_range(len(Episode_Rewards))
    y_Success= running_mean(Reached_Goal,average_over)
    plt.plot(x,y_Success)
    plt.xlabel('Episoden')
    plt.ylabel('Durchschnittliche Erfolgsrate (100 Episoden)')
    plt.grid(True)
    plt.xlim((average_over, len(Episode)))
    plt.ylim((0, 1))
    plt.savefig(Trainingname+'_Average_Success_Rate.png')
    plt.close()
    
    summe=0
    for i in range(len(Episode_Steps)):
        summe +=Episode_Steps[i]
    print ("Summe der Steps die gepolottet werden: "+str(summe))
    print ("Summe der Episoden die gepolottet werden: "+str(len(Episode)))
    summe2=0
    for i in range(len(Episode_Steps_initial)):
        summe2 +=Episode_Steps_initial[i]
    print ("Summe der Steps vor Reshape: "+str(summe2))
    print ("Summe der Episoden vor reshape: "+str(len(Episode_initial)))