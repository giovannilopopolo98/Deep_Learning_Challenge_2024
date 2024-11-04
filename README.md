# Deep_Learning_Challenge_2024

La challenge di deep learning viene usata come progetto valutativo. Essa richiede la stima della profondità in ogni pixel di un'immagine RGB per generare una DepthMap, con una codifica colore per la distanza dagli oggetti. Il professore ha assegnato un dataset per il progetto di stima della profondità monoculare, chiamato DepthEstimationUnreal, che include immagini RGB come input e ground truth di profondità come supervisione. Il dataset è organizzato in due sottocartelle: RGB (contenente immagini .jpg) e depth (contenente file di profondità .npy). Ciascuna di queste cartelle ha tre sottocartelle per train, validation e test (la cartella test è vuota e sarà utilizzata solo per la valutazione finale del modello, senza accesso ai dati). 

Nel template del progetto di stima della profondità monoculare, sono presenti le seguenti cartelle e file: 
- una cartella 'checkpoint' per salvare i checkpoint del modello;
- i file 'dataset.py', 'main.py' e 'utils.py', che sono implementati e modificabili solo nei parametri di 'main.py'.

Il file 'main.py' fornito dal professore contiene vari parametri modificabili per il progetto, come:
- il learning rate ('lr'),
- batch size,
- numero massimo di epoche ('max_epochs'),
- directory per i checkpoint ('ckpt_dir')
- directory dei dati ('data_dir').
