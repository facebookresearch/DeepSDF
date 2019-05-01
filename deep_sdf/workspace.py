#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch

model_params_subdir = 'ModelParameters'
optimizer_params_subdir = 'OptimizerParameters'
latent_codes_subdir = 'LatentCodes'
logs_filename = 'Logs.pth'
reconstructions_subdir = 'Reconstructions'
meshes_subdir = 'Meshes'
specifications_filename = 'specs.json'

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception('The experiment directory ({}) does not include specifications file ' + \
            '"specs.json"'.format(experiment_directory))

    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(experiment_directory, model_params_subdir,
        checkpoint + '.pth')

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data['model_state_dict'])


def build_decoder(experiment_directory, experiment_specs):

    arch = __import__('nnets.' + experiment_specs['NetworkArch'],
        fromlist=['Decoder'])

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(
                latent_size,
                **experiment_specs['NetworkSpecs']
            ).cuda()

    return decoder

def load_decoder(experiment_directory, experiment_specs,
                 checkpoint, data_parallel=True):

    decoder = build_decoder(experiment_directory, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    load_model_parameters(experiment_directory, checkpoint, decoder)

    return decoder

def load_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(experiment_directory, latent_codes_subdir, checkpoint + '.pth')

    if not os.path.isfile(filename):
        raise Exception('The experiment directory ({}) does not include a latent code file ' +
            'for checkpoint "{}"'.format(experiment_directory, checkpoint))

    data = torch.load(filename)

    num_vecs = data['latent_codes'].size()[0]
    latent_size = data['latent_codes'].size()[2]

    lat_vecs = []
    for i in range(num_vecs):
        lat_vecs.append(data['latent_codes'][i].cuda())

    return lat_vecs
