require 'nokogiri'
require 'open-uri'

page_of_white_people = "http://burningman.org/network/about-us/people/board-of-directors/"
puts "fetching page..."
doc = Nokogiri::HTML(open(page_of_white_people))
img_urls = doc.css(".bm-bio img").collect{|img| img.attributes["src"].value}
puts "#{img_urls.length} found."
img_urls.each{|img_url| `cd images;wget #{img_url}`}