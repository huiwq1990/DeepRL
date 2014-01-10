
default: mlp

random: experiment
	python2 random_agent.py

mlp: experiment
	python2 mlp_agent.py

experiment:
	rl_glue &
	#java -jar rl-library/products/MountainCar.jar &
	java -jar rl-library/products/CartPole.jar &
	#java -jar rl-library/products/Acrobot.jar &
	python2 experiment.py &

kill:
	ps | grep python2 | cut -f1 -d' ' | xargs kill
	ps | grep rl_glue | cut -f1 -d' ' | xargs kill

