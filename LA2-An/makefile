FCC = mpicc # tau_cc.sh 
LDC = mpicc # tau_cc.sh 
LD_FLAGS = -std=c11
FLAGS = -std=c11
PROG = mvp-student.cx
RM = /bin/rm
OBJS = mvp-student.o

# all rule
all: $(PROG)

# Add -lm for the imported math class
$(PROG) : $(OBJS)
	$(LDC) $(LD_FLAGS) $(OBJS) -o $(PROG)  -lm

%.o : %.c
	$(FCC) $(FLAGS) -c $<

# clean rule
clean:
	$(RM) -rf *.o $(PROG) *.mod run.o*