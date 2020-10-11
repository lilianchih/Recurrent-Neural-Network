rnn:
	g++ -fopenmp -O3 -isystem/usr/include/eigen3/ recurrentNetwork.cpp -o rnn
run:
	time ./rnn
clean:
	rm *.o
