#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import glob
import json
import logging
import os
import subprocess

import deep_sdf
import deep_sdf.workspace as ws

def filter_classes_glob(patterns, classes):
  import fnmatch

  passed_classes = set()
  for pattern in patterns:

    rule = lambda x: fnmatch.fnmatch(x, pattern)
    passed_classes = passed_classes.union(set(filter(rule, classes)))

  return list(passed_classes)

def filter_classes_regex(patterns, classes):
  import re

  passed_classes = set()
  for pattern in patterns:
    regex = re.compile(pattern)
    passed_classes = passed_classes.union(set(filter(regex.match, classes)))

  return list(passed_classes)

def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)

class NoMeshFileError(RuntimeError):
  """Raised when a mesh file is not found in a shape directory"""
  pass

class MultipleMeshFileError(RuntimeError):
  """"Raised when a there a multiple mesh files in a shape directory"""
  pass

def get_mesh_filename(shape_dir):
  mesh_filenames = list(glob.iglob(shape_dir + '/**/*.obj'))
  if len(mesh_filenames) == 0:
    return NoMeshFileError()
  elif len(mesh_filenames) > 1:
    raise MultipleMeshFileError()
  return mesh_filenames[0]

def process_mesh(mesh_filepath, target_filepath):
  logging.info(mesh_filepath + " --> " + target_filepath)
  executable = "bin/PreprocessMesh"
  subproc = subprocess.Popen([executable, "-m", mesh_filepath, "-o", target_filepath],
    stdout=subprocess.DEVNULL) #, stderr=subprocess.DEVNULL)
  subproc.wait()

def append_data_source_map(data_dir, name, source):

    data_source_map_filename = os.path.join(data_dir, '.data_sources.json');

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, 'r') as f:
            data_source_map = json.load(f);

    data_source_map[name] = os.path.abspath(source)

    with open(data_source_map_filename, 'w') as f:
        json.dump(data_source_map, f, indent=2)

    import sys
    sys.exit()

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=
      'Pre-processes data from a data source and append the results to a dataset.')
    arg_parser.add_argument('--data_dir', '-d', dest='data_dir', required=True, help=
      'The directory which holds all preprocessed data.')
    arg_parser.add_argument('--source', '-s', dest='source_dir', required=True, help=
      'The directory which holds the data to preprocess and append.')
    arg_parser.add_argument('--name', '-n', dest='source_name', default=None, help=
      'The name to use for the data source. If unspecified, it defaults to the directory name.')
    arg_parser.add_argument('--classes', '-c', dest='class_patterns', default=None, nargs='+', help=
      'This flag takes as arguments a pattern type (either "glob" or "regex"), followed by a ' + \
      'variable number of patterns.\nExamples: --classes glob \'03*\' \'04*\'' + \
      '\n          --classes regex \\d+')
    arg_parser.add_argument('--skip', dest='skip', default=False, action='store_true', help=
      'If set, previously-processed shapes will be skipped')
    arg_parser.add_argument('--threads', dest='num_threads', default=8, help=
      'The number of threads to use to process the data.')

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    if args.source_name is None:
      args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    dest_dir = os.path.join(args.data_dir, args.source_name)

    logging.info('Preprocessing data from ' + args.source_dir + ' and placing the results in ' +
      dest_dir)

    if not os.path.isdir(dest_dir):
      os.mkdir(dest_dir)

    append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    class_directories = os.listdir(args.source_dir)

    if not args.class_patterns is None:
      class_directories = filter_classes(args.class_patterns, class_directories)

    logging.debug('Processing classes: ' + str(args.class_patterns))

    meshes_and_targets = []

    for class_dir in class_directories:
      class_path = os.path.join(args.source_dir, class_dir)
      instance_dirs = os.listdir(class_path)

      logging.debug('Processing ' + str(len(instance_dirs)) + ' instances of class ' + \
        class_dir)

      target_dir = os.path.join(dest_dir, class_dir)

      if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

      for instance_dir in instance_dirs:

        shape_dir = os.path.join(class_path, instance_dir)

        processed_filepath = os.path.join(target_dir, instance_dir + '.npz')
        if args.skip and os.path.isfile(processed_filepath):
          logging.debug("skipping " + processed_filepath)
          continue

        try:
          mesh_filename = get_mesh_filename(shape_dir)

          meshes_and_targets.append((os.path.join(shape_dir, mesh_filename), processed_filepath))

        except NoMeshFileError:
          logging.warning("No mesh found for instance " + instance_dir)
        except MultipleMeshFileError:
          logging.warning("Multiple meshes found for instance " + instance_dir)


    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:

      for mesh_filepath, target_filepath in meshes_and_targets:
        executor.submit(process_mesh, mesh_filepath, target_filepath)

      executor.shutdown()
