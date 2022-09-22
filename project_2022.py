from time import time, sleep
import argparse
from classifier import classifier
from os import listdir


def main():
    #Defining start time to measure total time taken by the vgg model
    start_time = time()

    #Defining get_input_args function to create and retrieve commands
    in_arg = get_input_args()

    #Defining petlabels dictioanry with entries of all pet labels
    petlabels_dic = get_pet_labels(in_arg.dir)

    #Defining results dictionary with the labels obtained from the classifier
    results_dic = classify_images(petlabels_dic, in_arg.arch, in_arg.dir)

    print(results_dic)

    #Adjusting results dictionary to provide info on images that are of dogs
    adjust_results_4isdog(results_dic, in_arg.dogfile)

    #Defining results stats to provide all stats of images including accuracy of the chosen
    #model
    resultstats_dic = result_stats(results_dic)

    print_results(resultstats_dic, results_dic, in_arg.arch, True, True)

    end_time = time()

    total_time = end_time - start_time

    print(f"Total time taken by chosen model: \n{int((total_time / 3600))}:{int((total_time / 60))}:"
          f"{int((total_time % 60))}")


def get_input_args():
    """
    Creating 3 command line arguments:
    dir - Path to the pet image files
    arch - Path to the chosen model
    dogfile - path to the text file containing all labels associated to dogs.
    :returns:
     parse_args() - data structure that stores the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="pet_images/", help="Folder with pet images")
    parser.add_argument("--arch", type=str, default='vgg', help="Chosen model")
    parser.add_argument('--dogfile', type=str, default='dognames.txt', help="Dog names file")

    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Retrieves pet labels from image names in the pet images folder and adds them to the pet labels
    dictionary.
    For ease of comparison, .split(_) is used to seperate dog names with space
     when reading in pet filenames before adding them to the petlabels dictionary
    Parameters:
     image_dir - The path to the folder of images.
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet label as
     value
    """

    petlabels_dic = dict()

    in_file = listdir(image_dir)

    for idx in range(0, len(in_file), 1):

        if in_file[idx][0] != ".":

            images = in_file[idx].split("_")

            pet_labels = ""

            for word in images:
                if word.isalpha():
                    pet_labels += word.lower() + " "

            pet_labels = pet_labels.strip()

            if in_file[idx] not in petlabels_dic:
                petlabels_dic[in_file[idx]] = pet_labels

            else:
                print("Warning! File already in directory.",
                        in_file[idx])

    return petlabels_dic



def classify_images(petlables_dic, model, image_dir):
    """Uses the classifier function to create classifier labels and compare
    the results with the petlabels obtained from the get_pet_labels function.
    The function uses the classifier() function defined in classifier.py within
     this function.
     Parameters:
         petlables_dic - Dictionary that contains pet image labels in the
         folder as the key and the pet labels as the value.
         model - the path to the model used for the classifier
         image_dir - the full path to the folder og images to be classified
         by the CNN models.
     Returns:
         results dic - Dictionary with pet image labels as in the folder as
         the key, and the pet labels [0] and comparisons between the pet labels
          and the labels returned by the Classifier [1] with values 1 (for a
          match between the two) and 0 (if no match was found) as values."""

    results_dic = dict()

    for key in petlables_dic:

        model_label = classifier(image_dir+key, model)

        model_label = model_label.strip()
        model_label = model_label.lower()

        truth = petlables_dic[key]

        found = model_label.find(truth)

        if found >= 0:
            if ( ( (found == 0) and len(truth) == len(model_label)) or
                    ( (found == 0 or model_label[found - 1] == " ") or
                        ( (found + len(truth) == len(model_label)) and
                            ( (model_label[found + len(truth):found+len(truth)+1]) in
                            (",",".") )
                        )
                    )
            ):
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 1]
            else:
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 0]
        else:
            results_dic[key] = [truth, model_label, 0]

    return results_dic

in_arg = get_input_args()

petlabels_dic = get_pet_labels(in_arg.dir)

results_dic = classify_images(petlabels_dic, in_arg.arch, in_arg.dir)



def adjust_results_4isdog(results_dic, dogfile):
    """The function compares the pet labels obtained from image names
    and the labels returned from the classifier() function to the dognames
    text file with a list of dog names. It determines if classifier correctly
    classifies dog images 'as dogs' or 'not a dog'.
    Parameters:
        results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
        dogfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files.
    """

    dognames_dict = dict()

    with open(dogfile, "r") as dog_names:

        in_file = dog_names.readline()

        while in_file != "":

            in_file = in_file.rstrip()
            in_file = in_file.lower()

            if in_file not in dognames_dict:
                dognames_dict[in_file] = 1
            else:
                print("Warning! File already in dictioanry",
                        in_file)

            in_file = dog_names.readline()

        for key in results_dic:

            if results_dic[key][0] in dognames_dict:

                if results_dic[key][1] in dognames_dict:
                    results_dic[key].extend([1, 1])

                else:
                    results_dic[key].extend([1, 0])

            else:

                if results_dic[key][1] in dognames_dict:
                    results_dic[key].extend([0, 1])

                else:
                    results_dic[key].extend([0, 0])

    return results_dic



def result_stats(results_dic):
    """Creates resultsstats_dictionary with calculated statistics of the results
    obtained from the chosen model to determine the accuracy of results. The
    statisitics are displayed as percentages or counts.
    Parameters:
        results_dic
    Returns:
        resultstats_dic - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value.
                     """

    resultstats_dic = dict()

    resultstats_dic['n_images'] = len(results_dic)
    resultstats_dic['n_dog_images'] = 0
    resultstats_dic['n_correct_breeds'] = 0
    resultstats_dic['n_correct_dogs'] = 0
    resultstats_dic['n_nondog_img'] = 0
    resultstats_dic['n_matches'] = 0

    for key in results_dic:

        if sum(results_dic[key][2:]) == 3:
            resultstats_dic['n_correct_breeds'] += 1

        if results_dic[key][4] == 1:
            resultstats_dic['n_dog_images'] += 1

        if results_dic[key][2] == 1:
            resultstats_dic['n_matches'] += 1

            if sum(results_dic[key][3:]) == 0:
                resultstats_dic['n_nondog_img'] += 1

        if results_dic[key][3] == 1 and results_dic[key][4] == 1:
            resultstats_dic['n_correct_dogs'] += 1

        if sum(results_dic[key][2:]) == 0:

            resultstats_dic['n_nondog_img'] += 1


    resultstats_dic['pct_correct_breeds'] = (resultstats_dic['n_correct_breeds'] /
                                            resultstats_dic['n_dog_images']) * 100

    resultstats_dic['pct_matches'] = (resultstats_dic['n_matches'] /
                                    resultstats_dic['n_images']) * 100

    resultstats_dic['pct_correct_dogs'] = (resultstats_dic['n_correct_dogs'] /
                                        resultstats_dic['n_images']) * 100

    return resultstats_dic


def print_results(resultstats_dic, results_dic, model, print_incorrect_breeds=False,
                      print_incorrect_dogs=False):

    print(f"-----------RESULT STATS FOR {model.upper()} MODEL---------------\n")
    print("\nTotal Images: ", resultstats_dic['n_images'], "\nDog images: ",
            resultstats_dic['n_dog_images'], "\nCorrect non-dog images: ",
            resultstats_dic['n_nondog_img'], "\nTotal machtes: ",
            resultstats_dic['n_matches'])

    print("\n\nPercantages statistics: ")

    for stat in resultstats_dic:
        if stat[0] == 'p':
            print(f"{stat}: {resultstats_dic[stat]}%")

    if (print_incorrect_breeds and resultstats_dic['n_correct_breeds']
    != resultstats_dic['n_dog_images']):

        print("\n\n-----Incorrect breeds include-----")

        for key in results_dic:

            if sum(results_dic[key][2:]) == 2:
                print("Label: ", results_dic[key][0],   "Classified as: ",
                        results_dic[key][1])

    if (print_incorrect_dogs and resultstats_dic['n_correct_dogs'] +
    resultstats_dic['n_nondog_img'] != resultstats_dic['n_images']):

        print("\n\n-----Incorrect dogs------")

        for key in results_dic:

            if sum(results_dic[key][3:]) == 1:
                print("Label: ", results_dic[key][0],    "Classified as: ",
                        results_dic[key][1])



if __name__ == '__main__':
    main()