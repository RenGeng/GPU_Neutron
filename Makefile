all:
	$(MAKE) -C Sequentiel/
	$(MAKE) -C GPU/
	$(MAKE) -C GPU/stockage_contigu
	$(MAKE) -C CPU/
	$(MAKE) -C CPU_GPU/
	$(MAKE) -C CPU_GPU/Stockage_contigu

clean:
	$(MAKE) clean -C Sequentiel/
	$(MAKE) clean -C GPU/
	$(MAKE) clean -C GPU/stockage_contigu
	$(MAKE) clean -C CPU/
	$(MAKE) clean -C CPU_GPU/
	$(MAKE) clean -C CPU_GPU/Stockage_contigu
