all:
	g++ -std=c++11 -D__NO_INLINE__ nettalk_7000_1024.cpp -O2 -o try.out
	g++ -std=c++11 -D__NO_INLINE__ nettalk.cpp -O2 -o a.out
	g++ -std=c++11 -D__NO_INLINE__ figure7.cpp -O2 -o draw.out

