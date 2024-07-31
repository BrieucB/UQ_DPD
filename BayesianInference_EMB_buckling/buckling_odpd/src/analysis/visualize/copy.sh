mkdir -p output
rm -r output/* 2>/dev/null

SOURCE_DIR="../../trj_eq"
DEST_DIR="output"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check if the destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory does not exist: $DEST_DIR"
    echo "Creating a directory"
    mkdir $DEST_DIR
fi


for SUBFOLDER in "$SOURCE_DIR"/*/; do
    # Check if the subfolder is indeed a directory
    if [ -d "$SUBFOLDER" ]; then
        # Get the last file in the alphabetically sorted subfolder
        LAST_FILE=$(ls -1 "$SUBFOLDER" | sort | tail -n 1)
        echo $LAST_FILE
	    cp $SUBFOLDER$LAST_FILE $DEST_DIR
	fi
done

cd $DEST_DIR
rm gas*
