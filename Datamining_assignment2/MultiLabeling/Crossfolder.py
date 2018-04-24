#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class TenFolders:
    __nfolders = 0
    folders = []
    test_folder = {}
    categories = []
    train_data_count = 0
    test_data_count = 0

    def __init__(self, nfolders):
        self.__nfolders = nfolders
        for i in range(nfolders):
            folder = {}
            folder["data"] = []
            folder["labels"] = []
            self.folders.append(folder)

        self.test_folder["data"] = []
        self.test_folder["labels"] = []

    def add_data(self, data):
        folder_index = self.train_data_count % self.__nfolders
        self.folders[folder_index]["data"].append(data)
        self.train_data_count += 1

    def add_labels(self, labels):
        folder_index = self.test_data_count % self.__nfolders
        self.folders[folder_index]["labels"].append(labels)
        self.test_data_count += 1

    def add_categories(self, categories):
        self.categories = categories

    def add_test_data(self, data):
        self.test_folder["data"].append(data)

    def add_test_labels(self, labels):
        self.test_folder["labels"].append(labels)

    def orgnize_crossfolder(self, test_folder_index):
        data_train = []
        labels_train = []
        data_test = []
        labels_test = []
        for index in range(self.__nfolders):
            if index == test_folder_index:
                data_test.extend(self.folders[index]["data"])
                labels_test.extend(self.folders[index]["labels"])
            else:
                data_train.extend(self.folders[index]["data"])
                labels_train.extend(self.folders[index]["labels"])

        return data_train, labels_train, data_test, labels_test

    def simple_test(self):
        data_train = []
        labels_train = []
        data_test = []
        labels_test = []
        for index in range(self.__nfolders):
            data_train.extend(self.folders[index]["data"])
            labels_train.extend(self.folders[index]["labels"])

        data_test = self.test_folder["data"]
        labels_test = self.test_folder["labels"]

        return data_train, labels_train, data_test, labels_test

    def get_nfolders(self):
        return self.__nfolders
