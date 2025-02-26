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


## Miglior modello
Il miglior modello trovato finora ha ottenuto:
Loss: 0.7666

Training set:
- RMSE on TRAIN : 0.579466572321883
- SSIM on TRAIN: 0.856304222290669

Validation set:
- RMSE on VALIDATION : 2.810505590940777
- SSIM on VALIDATION: 0.5213211429746527

Test set:
- RMSE on TEST : 2.218269580288937
- SSIM on TEST : 0.6457219939482847

Risultati visivi Train 
![Image](https://github.com/user-attachments/assets/3dd3364a-1fbd-4ebb-b821-313a2a79c7d6)

Risultati visivi Validation
![Image](https://github.com/user-attachments/assets/9445fda1-14bc-478c-b91d-4cffeec958fe)

Risultati visivi Test
![Image](https://github.com/user-attachments/assets/bef12426-cda5-43be-96d5-7be4d41ff4fa)
![Image](https://github.com/user-attachments/assets/2b6d408e-1a5a-4997-a3ce-245e468fea53)


I pesi ricavati dal modello possono essere richiesti, non pote do caricarli su Github essendo il file molto grande
