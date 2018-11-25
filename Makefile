all:
	$(MAKE) -C Sequentiel/
	$(MAKE) -C GPU/
	$(MAKE) -C CPU/
	$(MAKE) -C CPU_GPU/

clean:
	$(MAKE) clean -C Sequentiel/
	$(MAKE) clean -C GPU/
	$(MAKE) clean -C CPU/
	$(MAKE) clean -C CPU_GPU/
