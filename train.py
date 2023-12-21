import os
import glob
import socket
import logging

import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


logger = logging.getLogger()


def multigpu_graph_def(model, data, config, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(config.BATCH_SIZE)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            images, config, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            images, config, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    config = ng.Config('inpaint.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    # training data
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        fnames = f.read().splitlines()
    data = ng.data.DataFromFNames(
        fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
    images = data.data_pipeline(config.BATCH_SIZE)
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(
        images, config=config)
    # validation images
    if config.VAL:
        with open(config.DATA_FLIST[config.DATASET][1]) as f:
            val_fnames = f.read().splitlines()
        # progress monitor by visualizing static images
        for i in range(config.STATIC_VIEW_SIZE):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                static_images, config, name='static_view/%d' % i)
    # training settings
    lr = tf.compat.v1.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.compat.v1.constant_initializer(1e-4))
    d_optimizer = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    g_optimizer = d_optimizer
    # gradient processor
    if config.GRADIENT_CLIP:
        gradient_processor = lambda grad_var: (
            tf.compat.v1.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
            grad_var[1])
    else:
        gradient_processor = None
    # log dir
    log_prefix = 'model_logs/' + '_'.join([
        ng.date_uid(), socket.gethostname(), config.DATASET,
        'MASKED' if config.GAN_WITH_MASK else 'NORMAL',
        config.GAN,config.LOG_DIR])
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=5,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'd'},
    )
    # train generator with primary trainer
    trainer = ng.train.Trainer(
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=config.MAX_ITERS,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs={
            'model': model, 'data': data, 'config': config, 'loss_type': 'g'},
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
    )
    # add all callbacks
    if not config.PRETRAIN_COARSE_NETWORK:
        trainer.add_callbacks(discriminator_training_callback)
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.compat.v1.summary.merge_all()),
    ])
    # launch training
    trainer.train()

    """
Antwort von Bing:

Der Code besteht aus mehreren Teilen:

    Der erste Teil liest die Konfigurationsdatei inpaint.yml, die verschiedene Parameter für das Modell und die Daten enthält. Er überprüft auch, ob eine GPU verfügbar ist, 
    und weist sie dem Modell zu.

    Der zweite Teil lädt die Trainingsdaten aus einer Liste von Dateinamen, die in config.DATA_FLIST[config.DATASET][0] gespeichert sind.
    Er erstellt ein Datenobjekt, das die Bilder in die gewünschte Form bringt und eine zufällige Ausschnittsoperation durchführt, wenn config.RANDOM_CROP wahr ist. 
    Er erzeugt auch eine Datenpipeline, die die Bilder in Chargen an das Modell liefert.

    Der dritte Teil erstellt das Hauptmodell, indem er die InpaintCAModel-Klasse instanziiert und die Methode build_graph_with_losses aufruft. 
    Diese Methode baut den Berechnungsgraphen für den Generator und den Diskriminator auf und definiert die Verlustfunktionen für beide. Der Generator verwendet eine Kombination 
    aus L1-, AE- und GAN-Verlusten, während der Diskriminator einen WGAN-GP-Verlust verwendet. Die Methode gibt die Variablen und die Verluste für beide Netzwerke zurück.

    Der vierte Teil lädt die Validierungsdaten aus einer anderen Liste von Dateinamen, die in config.DATA_FLIST[config.DATASET][1] gespeichert sind. 
    Er erstellt ein weiteres Datenobjekt und eine Datenpipeline für die Validierungsdaten. Er ruft auch die Methode build_static_infer_graph auf, die den Inferenzgraphen 
    für den Generator aufbaut und statische Bilder erzeugt, die den Fortschritt des Modells visualisieren.

    Der fünfte Teil definiert die Trainingseinstellungen, wie die Lernrate, die Optimierer, den Gradientenprozessor, das Log-Verzeichnis und die Gradienten-Clipping-Option.
    Er verwendet den Adam-Optimierer für beide Netzwerke und schneidet die Gradienten ab, wenn config.GRADIENT_CLIP wahr ist.

    Der sechste Teil erstellt zwei Trainer-Objekte, einen für den Generator und einen für den Diskriminator. Er verwendet die Trainer- und SecondaryTrainer-Klassen 
    aus der neuralgym-Bibliothek, die die Trainingsschleife und die Rückrufe verwalten. Er fügt verschiedene Rückrufe hinzu, wie Gewichts-Anzeiger, Modell-Wiederhersteller, 
    Modell-Speicher und Zusammenfassungs-Schreiber, die das Training überwachen und das Modell speichern.

    Der letzte Teil startet das Training, indem er die Methode train des Trainer-Objekts für den Generator aufruft. Diese Methode führt die Trainingsschleife aus,
    aktualisiert die Variablen, berechnet die Verluste und ruft die Rückrufe auf.


"""

"""
Copilot:

Here's a step-by-step breakdown of the script:

1. Import necessary modules: The script starts by importing necessary modules such as os, glob, socket, logging, tensorflow, neuralgym (a library for training neural networks),
and the InpaintCAModel class.

2. Define multigpu_graph_def function: This function is used to build the computation graph for the model on multiple GPUs. It takes as input the model, data, configuration, 
GPU ID, and loss type. It returns the loss for the generator or discriminator based on the loss_type argument.

3. Main execution: If the script is run as the main program, it starts by reading the configuration from inpaint.yml using ng.Config. It then sets the GPU(s) to be used for training.

4. Load training data: The script reads a list of file names from a file specified in the configuration. It then creates a data pipeline using ng.data.DataFromFNames
which loads and preprocesses the images.

5. Build the model: The script creates an instance of InpaintCAModel and builds the computation graph with losses using the build_graph_with_losses method.

6. Load validation data: If validation is enabled in the configuration, the script reads a list of validation file names and creates a data pipeline for them. 
It also builds a static inference graph for visualizing progress during training.

7. Set up training settings: The script sets up the learning rate, optimizers, and gradient processor. If gradient clipping is enabled in the configuration,
it sets up a function to clip gradients by average norm.

8. Prepare for logging: The script sets up a directory for logging training progress.

9. Set up training callbacks: The script sets up a secondary trainer for training the discriminator and a primary trainer for training the generator. 
It also adds callbacks for viewing weights, restoring and saving the model, and writing summaries for TensorBoard.

10. Start training: Finally, the script starts the training process by calling the train method of the trainer.

This script is a typical example of a training script for a GAN model in TensorFlow. It demonstrates how to set up data pipelines, build a model, set up training settings, 
and start the training process.
"""
