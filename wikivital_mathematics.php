<?php
$file = "wikivital_mathematics.json";
$data = file_get_contents($file);
$obj = json_decode($data, true);
// print_r($obj);

print_r(array_keys($obj));
// [0] => edges
// [1] => weights
// [2] => node_ids
// [3] => time_periods

print count($obj["edges"]);
print "\n";
print count($obj["weights"]);
print "\n";
print count($obj["node_ids"]);
print "\n";

print_r(array_keys($obj["node_ids"]));
foreach(array_keys($obj["node_ids"]) as $key) {
	print $key . "\n";
}

print $obj["time_periods"];
print "\n";
