# Total number of images in the Fitzpatrick17k dataset.
TOTAL_FITZPATRICK17K_IMAGES = 16577

# Use fastdup similarity scores to remove duplicates.
USE_SIMILARITY_THRESHOLD = True
# Use both annotators' labels to remove duplicates.
USE_A2_ANNOTATIONS = True
# Upper and lower similarity thresholds for removing duplicates.
SIMILARITY_THRESHOLD_UPPER = 0.99
SIMILARITY_THRESHOLD_LOWER = 0.7

# Use fastdup and cleanlab analyses to remove connected components of duplicates.
REMOVE_CC_DUPLICATES_FASTDUP_CLEANLAB = True

# Policy for removing clusters of duplicates.
# KeepOne (default): Keep one image from each cluster of duplicates.
# All: Remove all images from clusters of duplicates.
REMOVE_CC_DUPLICATES_LEVEL = "KeepOne"  # "KeepOne" (default) or "All"

# Use fastdup analysis to remove outliers.
REMOVE_OUTLIERS = True
# The outlier similarity threshold values from fastdup range from 0.376738 to 0.694547.
# Whatever threshold is specified here, the outliers with similarity values less than
# or equal to that threshold will be removed. If None, then all the outliers will be
# removed.
# None (default) or any value between 0.37 and 0.7.
OUTLIER_SIMILARITY_THRESHOLD = None  # None (default) or any value between 0.37 and 0.7.

# Remove images with missing FST labels.
# This is useful for removing images that do not have an FST label (denoted by `f0`).
# The default value is False because even images without an FST label are useful for
# training diagnostic models.
REMOVE_MISSING_FST = False

SAVE_FILTERED_FILE_LISTS = True
