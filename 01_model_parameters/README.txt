#mouse adding, 显示的时候，multidalliance_mouse.html
d.put("ENCSR940JVJ",["miRNA-Seq","4-1","ENCODE B6NCrl Fetal Skeletal Muscle miRNA-Seq"])
d.put("GSM1618365",["miRNA-Seq","4-2", "GEO C57BL/6J Myoblast (old mouse) miRNA-Seq"])
d.put("GSM1618369",["miRNA-Seq","4-2", "GEO C57BL/6J Myoblast (young mouse) miRNA-Seq"])
      d.put("add_atac_15", ["ATAC-Seq","8-1","ENCODE Embro 15.5 day limb tissue ATAC-Seq"])
      d.put("add_atac_14", ["ATAC-Seq","8-2","ENCODE Embro 14.5 day limb tissue ATAC-Seq"])
      d.put("add_atac_13", ["ATAC-Seq","8-3","ENCODE Embro 13.5 day limb tissue ATAC-Seq"])
      d.put("add_atac_12", ["ATAC-Seq","8-4","ENCODE Embro 12.5 day limb tissue ATAC-Seq"])
      d.put("add_atac_11", ["ATAC-Seq","8-5","ENCODE Embro 11.5 day limb tissue ATAC-Seq"])
      d.put("add_atac_c57", ["ATAC-Seq","8-6","GEO C57BL/6 Myoblast ATAC-Seq"])
      d.put("add_atac_c2mt_rep2",["ATAC-Seq","8-10","GEO C2C12 DM 72h ATAC-Seq"])
      d.put("add_atac_c2mb_rep2",["ATAC-Seq","8-8","GEO C2C12 Myoblast ATAC-Seq"])
      d.put("GSM721309",["MNase-Seq","9-2","GEO C2C12 DM 96h MNase-Seq"])
      d.put("c2c12dm72_g_1",["MNase-Seq","9-3","GEO C2C12 DM 72h MNase-Seq"])
      d.put("GSM721308",["MNase-Seq","9-1","GEO C2C12 Myoblast MNase-Seq"])


#用于上方piechart生成， in multidalliance_mouse.html
          d1.put("4-1", ["B6NCrl Fetal Skeletal Muscle, 1 track",1]);
          d1.put("4-2", ["C57BL/6J Myoblast, 2 tracks",2]);
          d1.put("8-1", ["Embryo 15.5d limb tissue, 1 track", 1])
          d1.put("8-2", ["Embryo 14.5d limb tissue, 1 track", 1])
          d1.put("8-3", ["Embryo 13.5d limb tissue, 1 track", 1])
          d1.put("8-4", ["Embryo 12.5d limb tissue, 1 track", 1])
          d1.put("8-5", ["Embryo 11.5d limb tissue, 1 track", 1])
          d1.put("8-6", ["C57BL/6 Myoblast, 1 track", 1])
          d1.put("8-10", ["C2C12 Myotube 72h, 1 track", 1])
          d1.put("8-8", ["C2C12 Myoblast, 1 track", 1])
          d1.put("9-2",["C2C12 Myotube 96h, 1 track", 1])
          d1.put("9-3",["C2C12 Myotube 72h, 1 track", 1])
          d1.put("9-1",["C2C12 Myoblast, 1 track", 1])


###############################################################



Query.html, entry1加的: (为了上方饼图显示)

        d.put("8-1", ["Embryonic 15.5d limb tissue, 1 dataset", 1])
        d.put("8-2", ["Embryonic 14.5d limb tissue, 1 dataset", 1])
        d.put("8-3", ["Embryonic 13.5d limb tissue, 1 dataset", 1])
        d.put("8-4", ["Embryonic 12.5d limb tissue, 1 dataset", 1])
        d.put("8-5", ["Embryonic 11.5d limb tissue, 1 dataset", 1])
        d.put("8-6", ["C57BL/6 Myoblast, 1 dataset", 1])
        d.put("8-10", ["C2C12 Myotube 72h, 1 dataset", 1])
        d.put("8-8", ["C2C12 Myoblast, 1 dataset", 1])
        d.put("9-1",["C2C12 MB, 2 datasets", 2])
        d.put("9-2",["C2C12 MT 96h, 1 dataset", 1])
        d.put("9-3",["C2C12 MT 72h, 1 dataset", 1])

 

explore 加的：
        d.put("4-1", ["B6NCrl Fetal Muscle","ENCSR940JVJ"]);
        d.put("4-2", ["C57BL/6J Myoblast", "GSM1618365/GSM1618369"]);
        d.put("5-1", ["C57BL/6J Muscle", "ENCSR000CNX"]);
        d.put("6-1", ["C2C12 Myoblast", "NOTPROVIDED"]);
        d.put("7-1", ["B6NCrl Fetal Muscle", "NOTPROVIDED"]);
        d.put("8-1", ["Embryonic 15.5d limb tissue", "add_atac_15"])
        d.put("8-2", ["Embryonic 14.5d limb tissue", "add_atac_14"])
        d.put("8-3", ["Embryonic 13.5d limb tissue", "add_atac_13"])
        d.put("8-4", ["Embryonic 12.5d limb tissue", "add_atac_12"])
        d.put("8-5", ["Embryonic 11.5d limb tissue", "add_atac_11"])
        d.put("8-6", ["C57BL/6 Myoblast", "add_atac_c57"])
        d.put("8-10", ["C2C12 Myotube 72h", "add_atac_c2mt_rep2"])
        d.put("8-8", ["C2C12 Myoblast", "add_atac_c2mb_rep2"])
                d.put("9-1",["C2C12 Myoblast", "GSM721308"])
        d.put("9-2",["C2C12 Myotube 96h", "GSM721309"])
        d.put("9-3",["C2C12 Myotube 72h","c2c12dm72_g_1"])
