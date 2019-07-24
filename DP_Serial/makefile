CC=gcc
PROG=work.x
OBJS=work.o
LD_FLAGS=
FLAGS=
RM=/bin/rm

all: $(PROG)

$(PROG):$(OBJS)
	$(CC) $(LD_FLAGS) $^ -o $@

%.o:%.c
	$(CC) -c $<

clean:
	rm -f $(OBJS) $(PROG)
