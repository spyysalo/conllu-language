# conllu-language

Language identification for CoNLL-U data

## Quickstart / example

Add language tags

```
mkdir identified
for f in examples/*; do
    python3 conllulang.py --use langid --limit-langs --threshold 0.9 $f \
        > identified/`basename $f`
done
```

Compare

```
diff -r examples identified
```

Eval

```
for f in identified/*; do
    echo $(basename $f) \
        $(egrep '^# language = ' $f | sort | uniq -c | sort -rn | head -n 1)
done
```
