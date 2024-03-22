import json

with open('imagenet.json', 'r') as f:
    labels_dict = json.load(f)

part_1 = 'линь, золотая рыбка, большая белая акула, тигровая акула, акула молот, электрический скат, морской скат, петух, курица, страус, ежевика, щегол, домашний зяблик, джунко, индиго_бантинг, малиновка, бюльбюль, сойка, сорока, синица, водный_узел, коршун, белоголовый орел, гриф, большая серая птица, европейская огненная саламандра, обыкновенная сова, eft, пятнистый саламандр, аксолотль, лягушка-бык, древесная лягушка, хвостатая лягушка, логгерхед, кожистая черепаха, замыкающаяся черепаха, бугорчатая черепаха, коробчатая черепаха, полосатый геккон, обыкновенная игуана, американский хамелеон, хлыстохвост, агама, оборчатый ящер, аллигатор, Гила_монстр, зеленый ящер, африканский хамелеон, Комодский дракон, Африканский_крокодил, американский_аллигатор, трицератопс, громовая змея, кольчатоголовая змея, когтистая змея, зеленая змея, королевская змея, подвязочная змея, водяная змея, виноградная змея, ночная змея, удав-констриктор, скальный питон, индийская кобра, зеленая мамба, морская змея, рогатая гадюка, алмазная спина, сайдвиндер, трилобит, жнец, скорпион, черный и золотой садовый паук, амбарный паук, садовый паук, черная вдова, тарантул, волчий паук, клещ, сороконожка, тетерев, куропатка, рябчик, степной цыпленок, павлин, перепел, куропатка, африканский серый попугай, ара, серохохлый какаду, лорикет, кукаль, пчелоед, птица-носорог, колибри, джакамар, тукан, селезень, красногрудая_мергансер, гусь, черный_сван, клыкастый, ехидна, утконос, валлаби, коала, вомбат, медуза, морская анемона, мозговая_кораль, плоский червь, нематода, раковина, улитка, слизень, морской_слуг, хитон, камерный_наутилус, подземный_краб, скальный_краб, скрипач_краб, королевский краб, американский лобстер, спинный_лобстер, раки, краб-отшельник, изопод, белый_исторк, черный_исторк, колпица, фламинго, маленький_синий_герон, американская цапля, выпь, журавль, лимпкин, европейская галлинула, американский кулик, дрофа, красноперка, красноспинный кулик, красношейка, донник, устричник, пеликан, королевский пингвин, альбатрос, серый кит, кит-убийца, дюгонь, морской лев, чихуахуа, японский спаниель, мальтийская собака, пекинес, ши-тцу, бленхеймский спаниель, папийон, тойтерьер, родезийский риджбек, афганская гончая, бассет, бигль, бладхаунд, блютик, черно-подпалая енотовидная собака, Ходячая гончая , Английская_фоксхаунд, рыжебородая, борзая, Ирландская_волфхаунд, Итальянская_грейхаунд, уиппет, Ибицкая гончая, норвежская элкхаунд, оттерхаунд, Салюки, скоттиш-дирхаунд, веймаранер, стаффордширский бультерьер, американский стаффордширский терьер, Бедлингтонский терьер, Бордертерьер, Керри-Блю-терьер, Ирландский терьер, Норфолкский терьер, Норвич_терьер, йоркшир_терьер, жесткошерстный_фокс_терьер, Лейкленд_терьер, Силихам_терьер, эрдельтерьер, кэрн, австралийский_терьер, Денди_динмонт, Бостон_буль, миниатюрный_шнауцер, великан_шнауцер, стандарт_шнауцер, шотландский_терьер, тибетский_терьер, шелковистый_терьер, мягкошерстный_терьер, Западно-высокогорный белый терьер, Лхаса, плоскошерстный ретривер, кудрявый ретривер, золотистый ретривер, лабрадорский ретривер, Чесапикский байский ретривер, немецкий короткошерстный ретривер, визсла, Английский_сеттер, Ирландский_сеттер, Гордон_сеттер, Бриттани_спаниель, кламбер, Английский_спрингер, Уэльский_спрингер_спаниель, кокер_спаниель, Сассекс_спаниель, ирландский_водный_спаниель, куваш, шипперке, гренендаль, малинуа, бриар, келпи, комондор, староанглийская овчарка, Шетландская овчарка, колли, бордер_колли, Бувье-де-Фландрский, ротвейлер, немецкая овчарка, доберман, миниатюрный пинчер, Большой_свисс_горный пес, бернский_горный пес, аппенцеллер, энтлебухер, боксер, бульмастиф, тибетский мастиф, французский бульдог, Грейт_дан, Сент_бернард, эскимосский пес, маламут, Сибирский хаски, далматин, аффенпинчер, басенджи, мопс, Леонберг, Ньюфаундленд, Грейт-Пиренеи, самоед, померанский шпиц, чау-чау, кеесхонд, брабансонский грифон, Пемброк, кардиган, игрушечный пудель, миниатюрный пудель, стандартный пудель, мексиканский бесшерстный, лесной волк, белый волк, рыжий волк, койот, динго, дол, африканская охотничья собака, гиена, рыжая лиса, китовая лиса, арктическая лиса, серая лиса, полосатая кошка, тигровая кошка, персидская кошка, сиамская кошка, египетская кошка, пума , рысь, леопард, снежный леопард, ягуар, лев, тигр, гепард, бурый медведь, американский черный медведь, ледяной медведь, ленивец, мангуст, сурикат, тигровая жучка, божья коровка, земляная жучка, длиннорогая жучка, листовая жучка, навозная жучка, носорог, долгоносик, муха, пчела, муравей, кузнечик, сверчок, ходячая палочка, таракан, богомол, цикада, кузнечик, златоглазка, стрекоза, бабочка, адмирал, колечко, монарх, капустная бабочка, серная бабочка, ликенида, морская звезда, морской еж, морской огурчик, лесной кролик, заяц, ангора, хомяк, дикобраз, лисица, сурок, бобр, гвинейская свинья, щавель, зебра, кабан, дикий кабан, бородавочник, бегемот, бык, водяной буффало , бизон, баран, снежный рог, горный козел, антилопа хартебек, импала, газель, арабская камель, лама, ласка, норка, хорек, черноногая феррет, выдра, скунс, барсук, броненосец, трехпалый ленивец, орангутанг, горилла, шимпанзе, гиббон, сиаманг, генон, патас, бабуин, макака, лангур, колобус, хоботный_монки, мартышка, капуцин, ревун_монки, тити, паук_монки, белка_монки, Мадагаскарская кошка, индри, индийский элефант, Африканский_элефант, меньшая_панда, гигантская_панда, барракута, угорь, кижуч, скалистая_красотка, рыба-анемон, осетр, гарь, крылатка, фугу, счеты, абайя, академическая_плата, аккордеон, акустическая_гитара, авианесущий аппарат, авиалайнер, дирижабль, алтарь, скорая помощь, амфибия, аналоговые_часы, пасека, фартук, пепельница, штурмовая винтовка, рюкзак, пекарня, балансировочная балка, воздушный шар, шариковая ручка, пластырь, банджо, перила, штанга, парикмахерское кресло, парикмахерская, сарай, барометр, бочка, тачка, бейсбол, баскетбол, люлька, фагот, шапочка для купания, банное полотенце, ванна, пляжная тележка, маяк, мензурка, медвежья шкура, пивная бутылка, пивной стакан, колокольчик, нагрудник, велосипед для двоих, бикини, переплет, бинокль, скворечник, лодочный сарай, бобслей, боло_тай, шляпка, книжный шкаф, книжный магазин, бутылочная крышка, бант, галстук-бабочка, латунь, бюстгальтер, волнорез, нагрудный знак, метла, ведро, пряжка, пуленепробиваемый жилет, bullet_train, мясная лавка, такси, котел, свеча, пушка, каноэ, открывалка для банок, кардиган, автомобильное зеркало, карусель, столярный набор, картонная коробка, автомобильное колесо, кассетный_машина, кассета, проигрыватель кассет, замок, катамаран, проигрыватель компакт-дисков, виолончель, сотовый телефон, цепочка, цепочка_защита, кольчуга, бензопила, сундук, шифоньер, колокольчик, фарфоровая шкатулка, рождественский чулок, церковь, кинотеатр, тесак, '
part_2 = 'скальные жилища, плащ, деревянные башмаки, шейкер для коктейлей, кофейная кружка, кофейник, катушка, комбинационный замок, компьютерная клавиатура, кондитерская, контейнерный корабль, трансформируемый, штопор, корнет, ковбойский ботинок, ковбойская шляпа, люлька, кран, аварийный шлем, ящик, детская кроватка, крокетный горшок, крокетный мяч, костыль, кираса, плотина, стол, настольный компьютер, дозвонщик, подгузник , цифровые часы, digital_watch, обеденный стол, тряпка для мытья посуды, посудомоечная машина, дисковый тормоз, док-станция, собачья упряжка, купол, коврик для двери, платформа для сверления, барабан, голень, гантель, голландская плита, электрический вентилятор, электрическая гитара, электрический локомотив, развлекательный центр, конверт, эспрессо_мастер, пудра для лица, боа из перьев, напильник, пожарная лодка, пожарный двигатель, пожарный экран, флагшток, флейта, складное кресло, футбольный шлем, вилочный погрузчик, фонтан, фонтан_пен, кровать с балдахином, грузовой автомобиль, французский рожок, сковородка для жарки, меховое пальто, мусоровоз, противогаз, газовый насос, кубок, картинг, гольф-мяч, тележка для гольфа, гондола, гонг, платье, grand_piano, теплица, решетка, бакалейный_магазин, гильотина, оползень для волос, лак для волос, половинная дорожка, молоток, корзина, ручная воздуходувка, ручной компьютер, носовой платок, жесткий диск, губная гармоника, арфа, комбайн, топор, кобура, домашний кинотеатр, соты, крючок, юбка-обруч, горизонтальная перекладина, лошадиная тележка, песочные часы, iPod, утюг, джек-фонарь, джинсы, джип, джерси, пазл-головоломка, джинрикиша, джойстик, кимоно, наколенник, узел, лабораторный халат, половник, абажур, ноутбук, газонокосилка, кепка для объектива, открывалка для писем, библиотека, спасательная шлюпка, зажигалка, лимузин, подводка, губная помада, мокасины, лосьон, громкоговоритель, лупа, лесопилка, магнитный_компас, сумка для почты, почтовый ящик, майо, майо, крышка люка, марака, маримба, маска, спичка, майское дерево, лабиринт, измерительная чашка, медицинский сундук, мегалит, микрофон, микроволновая печь, военная униформа, банка для молока, микроавтобус, мини-юбка, минивэн, ракета, рукавица, смеситель, мобильный дом, Модель_т, модем, монастырь, монитор, мопед, миномет, строительная доска, мечеть, москитная сетка, мотороллер, горный велосипед, маунтин_тент, мышь, мышеловка, движущийся фургон, намордник, гвоздь, шейный браслет, ожерелье, соска, записная книжка, обелиск, гобой, окарина, одометр, масляный фильтр, орган, осциллограф, верхняя юбка, воловья повозка, кислородная_маска, пакет, весло, гребное колесо, навесной замок, кисть, пижама, дворец, водосточная труба, бумажное полотенце, парашют, параллельные брусья, парковочный стол, парковочный счетчик, пассажирский автомобиль, патио, телефон-автомат, тумба, пенал для карандашей, точилка для карандашей, духи, Петри_диш, ксерокс, отмычка, пикелхауб, пикет_защита, самовывоз, пирс, копилка, бутылка для пилюль, подушка, мяч для пинг-понга, вертушка, пират, кувшин, самолет, планетарий, пластиковый пакет, подставка_ для тарелок, плуг, вантуз, полароидная камера, шест, полицейский фургон, пончо, бильярдный стол, поп-бутылка, горшок, гончарное колесо, power_drill, молитвенный круг, принтер, тюрьма, снаряд, проектор, шайба, сумка для битья, кошелек, перо, одеяло, гонщик, ракетка, радиатор, радио, радиотелескоп, дождевая бочка, транспортное средство для отдыха, катушка, зеркальная камера, холодильник, пульт дистанционного управления, ресторан, револьвер, винтовка, кресло-качалка, гриль, резиновая резинка, регби-мяч, правило, беговая обувь, сейф, безопасная булавка, солонка, сандалия, саронг, саксофон, ножны, весы, школьный автобус, шхуна, табло, ширма, винт, отвертка, ремень безопасности, швейная машина, щит, обувной магазин, седзи, корзина для покупок, тележка для покупок, лопата, шапочка для душа, занавеска для душа, лыжи, лыжная маска, спальный мешок, скользящая ручка, раздвижная дверь, прорезь, трубка, снегоход, снегоочиститель, мыльница, футбольный мяч, носок, солярий, сомбреро, чаша для супа, космический бар, космический подогреватель, космический челнок, лопатка, скоростной катер, паутина, веретено, спортивный автомобиль, прожектор, сцена, паровозик, стальной мост, стальная барабанная перепонка, стетоскоп, палантин, каменная стена, секундомер, плита, ситечко, трамвай, носилки, студийная кушетка, ступа, подводная лодка, костюм, солнечные часы, солнцезащитное стекло, солнцезащитные очки, солнцезащитный крем, подвесной мостик, тампон, толстовка, плавательные принадлежности, качели, выключатель, шприц, настольная лампа, резервуар, магнитофон, чайник, плюшевый мишка, телевизор, теннисный мяч, соломенная крыша, театральная занавеска, наперсток, молотилка, трон, облицовочный потолок, тостер, табачная лавка, сиденье для туалета, факел, тотемный столб, тягач, магазин игрушек, трактор, прицепной грузовик, поднос, плащ-палатка, трехколесный велосипед, тримаран, тренога, триумфальная арка, троллейбус, тромбон, ванна, турникет, клавиатура для пишущей машинки, зонт, одноколесный велосипед, стойка, пылесос, ваза, хранилище, бархат, торговая_машина, облачение, виадук, скрипка, волейбольный мяч, вафельный_ирон, настенные часы, кошелек, гардероб, военный самолет, умывальник, стиральная машина, бутылка для воды, кувшин для воды, водонапорная башня, кувшин для виски, свисток, парик, оконный экран, оконный абажур, Виндзорский галстук, бутылка для вина, крыло, вок, деревянная ложка, шерсть, червячная ограда, крушение, зевак, юрта, веб_сайт, комикс_бук, кроссворд_пузли, уличный_знак, светофор, книжный_жакет, меню, тарелка, гуакамоле, консоме, горячий_пот, трайфл, мороженое, фруктовый лед, французский_рулет, рогалик, крендель, чизбургер, хот-дог, картофельное пюре, кочанная капуста, брокколи, цветная капуста, цуккини, спагетти с мякотью, желуди с мякотью, орехи с мякотью, огурец, артишок, болгарский перец, кардамон, грибы, Бабушки_смит, клубника, апельсин, лимон, инжир, ананас, банан, джекфрут, заварное яблоко, гранат, сено, карбонара, шоколадное пюре, тесто, мясной рулет, пицца, запеканка, буррито, красное вино, эспрессо, чашка, гоголь-моголь, альп, пузырь, утес, коралловый риф, гейзер, берег озера, мыс, песчаная отмель, берег моря, долина, вулкан, игрок в мяч, жених, скуба_дивер, рапс, маргаритка, Башмачок мелкоцветковый, кукуруза, желудь, шиповник, бакай, коралловый грибок, мухомор, гиромитра, вонючий рог, земная звезда, лесная курица, подберезовик, ухо, туалетная ткань'

res = part_1 + part_2
res_list = res.split(', ')
for i in range(len(res_list)):
    labels_dict[str(i)] = res_list[i]
    #print(list(labels_dict.values())[i], '-------', res_list[i])
print(labels_dict)

with open('russian_imagenet.json', 'w') as f:
    # Сериализуем словарь и записываем его в файл в формате JSON
    json.dump(labels_dict, f, indent=4)

with open('russian_imagenet.json', 'r') as f:
    labels_dict = json.load(f)
    print(labels_dict)