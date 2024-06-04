### Projekt Własny - Sieć Fast-RCNN
Projekt miał na celu dostrojenie modelu dostępnego w bibliotece torch do detekcji obiektów na obrazach zbioru VOC.

Niestety okazało się, iż i model i zbiór są zbyt obszerne dla moich zasobów i dotrenowanie tej sieci wymagało dużego zbioru treningowego i testowego. To powodowało, że trenowanie jednej epoki trwało około 21 godzin na mojej maszynie.

W sprawozdaniu ująłem podstawową teorię związaną z moich początkowym podejściem oraz dwa zrzuty ekranu pokazujące działanie tej sieci dla bardzo małego zbioru danych (w porównaniu do modelu i zbioru danych).

Początkowe próby są zawarte w pliku main.py. Zbioru VOC nie zawarłem, gdyż ważył ponad 4GB.

Przez brak możliwości dotrenowania jej w żaden sposób oraz już brak czasu zdecydowałem się na zmianę podejścia i stworzyłem plik main2.py, w którym jest podstawowa sieć konwolucyjna klasyfikująca zbiór CIFAR10. Druga (krótsza) część sprawozdania posiada zrzuty ekranu posiadające wyniki dla zmian hiperparametrów tej sieci oraz podstawowe przemyślenia.
