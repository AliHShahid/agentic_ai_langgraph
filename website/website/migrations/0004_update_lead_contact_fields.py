from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0003_alter_lead_package'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lead',
            name='name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='lead',
            name='company',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='lead',
            name='package',
            field=models.CharField(choices=[('chatbots', 'AI Chatbots & RAG'), ('automation', 'Workflow Automation'), ('mobile', 'Mobile App Development'), ('consult', 'AI Consultation'), ('web', 'Web Development'), ('analytics', 'Data Analysis'), ('mldl', 'Machine Learning Tasks'), ('custom', 'Custom Request')], max_length=50),
        ),
        migrations.AlterField(
            model_name='lead',
            name='message',
            field=models.TextField(),
        ),
        migrations.AddField(
            model_name='lead',
            name='submitted_by',
            field=models.CharField(choices=[('web', 'Website Form'), ('chatbot', 'AI Chatbot')], default='web', max_length=50),
        ),
        migrations.AlterModelOptions(
            name='lead',
            options={'ordering': ['-created_at']},
        ),
    ]