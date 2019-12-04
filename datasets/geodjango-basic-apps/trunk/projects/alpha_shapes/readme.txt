This is a little sample app pulled together for a demo at #djangoseattle meetup

Thanks to humanhistory for the reminder that working with Aaron's data in geodjango would be cool.

Read all about Alpha Shapes here:

http://code.flickr.com/blog/tag/clustr/


Requires:

1) Geodjango and dependencies
2) Mapnik
3) Flickr data

To get up and running:

1) create a postgis enabled database
2) syncdb and all that
3) runserver and make sure the admin works
4) load some of the flickr data with the provided script
5) then view the site root / and see if some of the data loads (it is restricted by default to the first 200 records)
 - It is working if you see a small group of orange dots/shapes
6) Install mapnik and try to get the whole dataset rendered (turn on the Mapnik WMS layer in the upper right side of the map)
7) ## BE AFAID ## - then play around with the fragile query window which passes your text into a view and evals it - yikes!


Actually get Flickr data to use in this app via:

http://www.flickr.com/services/shapefiles/1.0/
gzip -d flickr_shapefiles_public_dataset_1.0.1.xml.gz

Import the data into this project with the 'load_flickr_data.py' script written by 'humanhistory' in #geodjango irc


You'll need to edit the path to the 'flickr_shapefiles_public_dataset_1.0.1.xml' file and ignore a bunch of errors.

NOTE: 'DEBUG=False' by default so this import does not hog memory 

I've yet to actually let the import script finish... let me know how far you get :)


