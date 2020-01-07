using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;


namespace MTGModelCreation
{
    class CardData
    {

        [LoadColumn(0)]
        public string Artist;
        
        [LoadColumn(1)]
        public float ConvertedManaCost;

        [LoadColumn(2)]
        public string Loyalty;

        [LoadColumn(3)]
        public string ManaCost;

        [LoadColumn(4)]
        public string Name;

        [LoadColumn(5)]
        public string Power;

        [LoadColumn(6)]
        public float Price;

        [LoadColumn(7)]
        public string Rarity;

        [LoadColumn(8)]
        public string Text;

        [LoadColumn(9)]
        public string Toughness;

        [LoadColumn(10)]
        public string Type;

    }
}
