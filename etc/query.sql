
show databases;

show tables;

create database smart_aquarium;

use smart_aquarium;

create table `data`(
`_id` int(10) not null auto_increment,
`coordinates_x` int(10) default null comment 'x',
`coordinates_y`int(10) default null comment 'y',
`depth` int(10) default null comment 'depth',
`timestamp` varchar(500) default null comment 'time stamp',
`frontCam_w` int(10) default null comment 'front camera width',
`frontCam_h` int(10) default null comment 'front camera height',
`sideCam_w` int(10) default null comment 'side camera width',
`sideCam_h` int(10) default null comment 'side camera height',
key `_id` (`_id`)
) engine=InnoDB default charset=utf8;


INSERT INTO `smart_aquarium`.`data` (
`_id`, `coordinates_x`, `coordinates_y`, `depth`, `timestamp`, `frontCam_w`, `frontCam_h`, `sideCam_w`, `sideCam_h`
) VALUES (
'1', '572', '84', '80', '154017038.7213247', '640', '480', '640', '480'
);

select * from data;

drop table data;

