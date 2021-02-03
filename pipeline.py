import torch
import pytorch_lightning as pl

from argparse import ArgumentParser

from components.network import Style_Encoder, Style_Decoder


class Baseline(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

    def __init__(self):
        pass
